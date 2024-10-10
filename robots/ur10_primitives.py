from pybullet_tools.pr2_utils import get_top_grasps
from pybullet_tools.utils import (
    get_movable_joints,
    set_joint_positions,
    link_from_name,
    sample_placement,
    end_effector_from_body,
    approach_from_grasp,
    plan_joint_motion,
    GraspInfo,
    Pose,
    Point,
    inverse_kinematics,
    pairwise_collision,
    get_sample_fn,
    plan_direct_joint_motion,
    wait_if_gui,
)
from robots.utils import (
    BodyPose,
    BodyGrasp,
    BodyConf,
    Attach,
    Detach,
    BodyPath,
    Command,
)
from itertools import count

def get_grasp_gen(robot, grasp_name="top"):
    """Return a generator that defines object grasps of a robot.

    Given an object, it will first be approximated as a cuboid (AABB),
    the function then generates potential grasp poses in object frame
    """
    # UR10 + robotis RH-P12-RN
    tool_link_name = "robot_ee_link"
    tool_link = link_from_name(robot, tool_link_name)
    tool_max_width = 0.106
    tool_offset = Pose(Point(z=-0.05))

    if grasp_name == "top":
        grasp_info = GraspInfo(
            # Generate top grasp poses in tool frame
            get_grasps=lambda body: get_top_grasps(
                body,
                under=True,  # symmetric grasp
                tool_pose=tool_offset,
                body_pose=Pose(),
                max_width=tool_max_width,
                grasp_length=0.0,
            ),
            # Approach pose for the grasp (top)
            approach_pose=Pose(0.1 * Point(z=1)),
        )
    else:
        raise ValueError("Invalid grasp name", grasp_name)

    def gen(body):
        grasp_poses = grasp_info.get_grasps(body)
        # TODO: continuous set of grasps
        for grasp_pose in grasp_poses:
            body_grasp = BodyGrasp(
                body, grasp_pose, grasp_info.approach_pose, robot, tool_link
            )
            yield (body_grasp,)

    return gen


def get_ik_fn(robot, fixed=[], teleport=False, num_attempts=10):
    """Return a IK function that finds an approach configuration
    for grasping an object with a given grasp.

    The function will ren approach turn a configuration and a command.
    The command starts at approach configuration, moves to grasp configuration,
    attach object, and then retreats to the approach configuration.
    """
    movable_joints = get_movable_joints(robot)
    gripper_joints = movable_joints[-2:]  # gripper joints
    movable_joints = movable_joints[:-2]  # remove gripper joints
    sample_fn = get_sample_fn(robot, movable_joints)  # sample configuration

    def fn(body, pose, grasp):
        # Obstacles: object + fixed environment objects
        obstacles = [body] + fixed

        # Compute desired grasp pose and approach pose in world frame
        gripper_pose = end_effector_from_body(pose.pose, grasp.grasp_pose)
        approach_pose = approach_from_grasp(grasp.approach_pose, gripper_pose)

        for _ in range(num_attempts):
            # Random seed
            set_joint_positions(robot, movable_joints, sample_fn())

            # Approach configuration
            # TODO: multiple attempts?
            q_approach = inverse_kinematics(robot, grasp.link, approach_pose)
            # remove gripper joints
            if q_approach is not None:
                q_approach = list(q_approach[:-2]) + [0, 0]

            if (q_approach is None) or any(
                pairwise_collision(robot, b) for b in obstacles
            ):
                continue
            conf = BodyConf(robot, q_approach)

            # Grasp configuration
            q_grasp = inverse_kinematics(robot, grasp.link, gripper_pose)

            # remove gripper joints
            if q_grasp is not None:
                q_grasp = list(q_grasp[:-2]) + [0, 0]

            if (q_grasp is None) or any(
                pairwise_collision(robot, b) for b in obstacles
            ):
                continue

            # TODO
            q_approach = q_approach[:-2]
            q_grasp = q_grasp[:-2]
            conf = BodyConf(robot, q_approach, joints=movable_joints)

            if teleport:
                path = [q_approach, q_grasp]

            else:
                conf.assign()
                # direction, _ = grasp.approach_pose
                # path = workspace_trajectory(robot, grasp.link, point_from_pose(approach_pose), -direction,
                #                                   quat_from_pose(approach_pose))
                path = plan_direct_joint_motion(
                    robot,
                    conf.joints,
                    q_grasp,
                    obstacles=obstacles,
                    disabled_collisions=set([(10, 12), (11, 13)]),
                )
                if path is None:
                    # if DEBUG_FAILURE:
                    #     wait_if_gui("Approach motion failed")
                    continue

            # TODO
            # Store results as a command
            command = Command(
                [
                    BodyPath(robot, path, joints=movable_joints),
                    Attach(body, robot, grasp.link, gripper_joints),
                    BodyPath(
                        robot,
                        path[::-1],
                        joints=movable_joints,
                        attachments=[grasp],
                    ),
                ]
            )
            return (conf, command)

            # TODO: holding collisions
        return None

    return fn


def get_free_motion_gen(robot, fixed=[], teleport=False, self_collisions=True):
    """Return a function that generates a collision free motion plan."""

    def fn(conf1, conf2, fluents=[]):
        assert (conf1.body == conf2.body) and (conf1.joints == conf2.joints)

        if teleport:
            path = [conf1.configuration, conf2.configuration]

        else:
            conf1.assign()
            obstacles = fixed + assign_fluent_state(fluents)
            # Plan a collision free path
            path = plan_joint_motion(
                robot,
                conf2.joints,
                conf2.configuration,
                obstacles=obstacles,
                self_collisions=self_collisions,
                disabled_collisions=set([(10, 12), (11, 13)]),
            )
            if path is None:
                return None

        # Store results as a command
        command = Command([BodyPath(robot, path, joints=conf2.joints)])
        return (command,)

    return fn


def get_holding_motion_gen(
    robot, fixed=[], teleport=False, self_collisions=True
):
    """Return a function that generates a collision free motion plan
    while holding an object.
    """

    def fn(conf1, conf2, body, grasp, fluents=[]):
        assert (conf1.body == conf2.body) and (conf1.joints == conf2.joints)

        if teleport:
            path = [conf1.configuration, conf2.configuration]

        else:
            conf1.assign()
            obstacles = fixed + assign_fluent_state(fluents)
            # Plan a collision free path
            path = plan_joint_motion(
                robot,
                conf2.joints,
                conf2.configuration,
                obstacles=obstacles,
                attachments=[grasp.attachment()],
                self_collisions=self_collisions,
                disabled_collisions=set([(10, 12), (11, 13)]),
            )
            if path is None:
                return None

        # Store results as a command
        command = Command(
            [BodyPath(robot, path, joints=conf2.joints, attachments=[grasp])]
        )
        return (command,)

    return fn


def get_movable_collision_test():
    def test(command, body, pose):
        if body in command.bodies():
            return False
        pose.assign()
        for path in command.body_paths:
            moving = path.bodies()
            if body in moving:
                # TODO: cannot collide with itself
                continue
            for _ in path.iterator():
                # TODO: could shuffle this
                if any(pairwise_collision(mov, body) for mov in moving):
                    if DEBUG_FAILURE:
                        wait_if_gui("Movable collision")
                    return True
        return False

    return test


def get_stable_gen(fixed=[]):
    def gen(body, surface):
        while True:
            pose = sample_placement(body, surface)
            if (pose is None) or any(
                pairwise_collision(body, b) for b in fixed
            ):
                continue
            body_pose = BodyPose(body, pose)
            yield (body_pose,)

    return gen


def assign_fluent_state(fluents):
    obstacles = []
    for fluent in fluents:
        name, args = fluent[0], fluent[1:]
        if name == "atpose":
            o, p = args
            obstacles.append(o)
            p.assign()
        else:
            raise ValueError(name)
    return obstacles
