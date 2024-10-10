"""Utils functions for robot primitives"""

import time
from itertools import count

from pybullet_tools.utils import (
    get_pose,
    set_pose,
    get_movable_joints,
    set_joint_positions,
    enable_real_time,
    disable_real_time,
    joint_controller,
    Attachment,
    step_simulation,
    refine_path,
    get_joint_positions,
    add_fixed_constraint,
    remove_fixed_constraint,
    wait_for_duration,
    wait_if_gui,
    flatten,
    joint_controller_hold,
    get_min_limit,
    get_max_limit,
)


class BodyPose(object):
    """A class to represent a potential pose of a body in the world.

    assign() can be used to set the stored pose to the body.
    """

    num = count()

    def __init__(self, body, pose=None):
        if pose is None:
            pose = get_pose(body)
        self.body = body
        self.pose = pose
        self.index = next(self.num)

    @property
    def value(self):
        return self.pose

    def assign(self):
        set_pose(self.body, self.pose)
        return self.pose

    def __repr__(self):
        index = self.index
        # index = id(self) % 1000
        return "p{}".format(index)


class BodyGrasp(object):
    """A class to represent a potential grasp of a body in the world.

    assign() can be used to set the stored grasp pose to the body,
    so that the body is attached to the link of the robot.
    """

    num = count()

    def __init__(self, body, grasp_pose, approach_pose, robot, link):
        self.body = body
        self.grasp_pose = grasp_pose  # relative to the link
        self.approach_pose = approach_pose  # relative to the link
        self.robot = robot
        self.link = link
        self.index = next(self.num)

    @property
    def value(self):
        return self.grasp_pose

    @property
    def approach(self):
        return self.approach_pose

    # def constraint(self):
    #     grasp_constraint()

    def attachment(self):
        return Attachment(self.robot, self.link, self.grasp_pose, self.body)

    def assign(self):
        return self.attachment().assign()

    def __repr__(self):
        index = self.index
        # index = id(self) % 1000
        return "g{}".format(index)


class BodyConf(object):
    """A class to represent a potential configuration of a robot in the world.

    assign() can be used to set the stored configuration to the robot.
    """

    num = count()

    def __init__(self, body, configuration=None, joints=None):
        if joints is None:
            joints = get_movable_joints(body)
        if configuration is None:
            configuration = get_joint_positions(body, joints)
        self.body = body
        self.joints = joints
        self.configuration = configuration
        self.index = next(self.num)

    @property
    def values(self):
        return self.configuration

    def assign(self):
        set_joint_positions(self.body, self.joints, self.configuration)
        return self.configuration

    def __repr__(self):
        index = self.index
        # index = id(self) % 1000
        return "q{}".format(index)


class ApplyForce(object):
    def __init__(self, body, robot, link, joints):
        self.body = body
        self.robot = robot
        self.link = link
        self.joints = joints

    def bodies(self):
        return {self.body, self.robot}

    def iterator(self, **kwargs):
        return []

    def refine(self, **kwargs):
        return self

    def __repr__(self):
        return "{}({},{})".format(
            self.__class__.__name__, self.robot, self.body
        )

    def control(self, dt, **kwargs):
        raise NotImplementedError

    def reverse(self):
        raise NotImplementedError


class Attach(ApplyForce):
    def control(self, dt=1 / 240.0, **kwargs):
        joints = self.joints
        values = [
            get_max_limit(self.robot, joint) for joint in joints
        ]  # Closed

        t = 0
        for _ in joint_controller_hold(self.robot, joints, values):
            step_simulation()
            time.sleep(dt)
            t = t + dt
            if t > 1:
                break
        # add_fixed_constraint(self.body, self.robot, self.link)

    def reverse(self):
        return Detach(self.body, self.robot, self.link, self.joints)


class Detach(ApplyForce):
    def control(self, dt=1 / 240.0, **kwargs):
        joints = self.joints
        values = [get_min_limit(self.robot, joint) for joint in joints]  # Open

        t = 0
        for _ in joint_controller_hold(self.robot, joints, values):
            step_simulation()
            time.sleep(dt)
            t = t + dt
            if t > 0.2:
                break
        # remove_fixed_constraint(self.body, self.robot, self.link)

    def reverse(self):
        return Attach(self.body, self.robot, self.link, self.joints)


class BodyPath(object):
    """A class to represent a path of configurations for a robot in the world.

    There are different ways to execute the path:
    - iterator() simulates one configuration at a time
    - control() executes the path in simulation
    """

    def __init__(self, body, path, joints=None, attachments=[]):
        if joints is None:
            joints = get_movable_joints(body)
        self.body = body
        self.path = path
        self.joints = joints
        self.attachments = attachments

    def bodies(self):
        return set(
            [self.body] + [attachment.body for attachment in self.attachments]
        )

    def iterator(self):
        """Simulate one configuration at a time."""
        # TODO: compute and cache these
        # TODO: compute bounding boxes as well
        for i, configuration in enumerate(self.path):
            set_joint_positions(self.body, self.joints, configuration)
            for grasp in self.attachments:
                grasp.assign()
            yield i

    # TODO: what is the purpose of having real_time here?
    def control(self, real_time=False, dt=1 / 240.0):
        """Execute the path in simulation."""
        # TODO: just waypoints
        if real_time:
            enable_real_time()
        else:
            disable_real_time()
        for values in self.path:
            for _ in joint_controller(
                self.body,
                self.joints,
                values,
                tolerance=5e-3,
                position_gain=0.5,
            ):
                if not real_time:
                    step_simulation()
                time.sleep(dt)

    # def full_path(self, q0=None):
    #     # TODO: could produce sequence of savers
    def refine(self, num_steps=0):
        """Refine the path by linear interpolation."""
        return self.__class__(
            self.body,
            refine_path(self.body, self.joints, self.path, num_steps),
            self.joints,
            self.attachments,
        )

    def reverse(self):
        return self.__class__(
            self.body, self.path[::-1], self.joints, self.attachments
        )

    def __repr__(self):
        return "{}({},{},{},{})".format(
            self.__class__.__name__,
            self.body,
            len(self.joints),
            len(self.path),
            len(self.attachments),
        )


class Command(object):
    """A class to represent a sequence of body paths for a robot in the world.

    There are different ways to execute the command:
    - step() simulates one configuration at a time, controlled by user gui
    - execute() simulates one configuration at a time, automatically
    - control() executes the path in simulation
    """

    num = count()

    def __init__(self, body_paths):
        self.body_paths = body_paths
        self.index = next(self.num)

    def bodies(self):
        return set(flatten(path.bodies() for path in self.body_paths))

    # def full_path(self, q0=None):
    #     if q0 is None:
    #         q0 = Conf(self.tree)
    #     new_path = [q0]
    #     for partial_path in self.body_paths:
    #         new_path += partial_path.full_path(new_path[-1])[1:]
    #     return new_path
    def step(self):
        """Simulate one configuration at a time, controlled by user gui."""
        for i, body_path in enumerate(self.body_paths):
            for j in body_path.iterator():
                msg = "{},{}) step?".format(i, j)
                wait_if_gui(msg)
                # print(msg)

    def execute(self, time_step=0.05):
        """Simulate one configuration at a time, automatically."""
        for i, body_path in enumerate(self.body_paths):
            for j in body_path.iterator():
                # time.sleep(time_step)
                wait_for_duration(time_step)

    def control(self, real_time=False, dt=1 / 240.0):
        """Execute the path in simulation."""
        for body_path in self.body_paths:
            body_path.control(real_time=real_time, dt=dt)

    def refine(self, **kwargs):
        """Refine the path by linear interpolation."""
        return self.__class__(
            [body_path.refine(**kwargs) for body_path in self.body_paths]
        )

    def reverse(self):
        return self.__class__(
            [body_path.reverse() for body_path in reversed(self.body_paths)]
        )

    def __repr__(self):
        index = self.index
        # index = id(self) % 1000
        return "c{}".format(index)
