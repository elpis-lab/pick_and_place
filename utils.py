import os
from pybullet_tools.utils import load_pybullet, set_pose
from itertools import count

def load_model(rel_path, pose=None, **kwargs):
    """Load a model from a relative path and set its pose if given."""
    model_dir = os.path.abspath(os.path.dirname(__file__))
    abs_path = os.path.join(model_dir, "models", rel_path)
    # If the file does not exist in default models folder,
    # use the relative path directly
    if not os.path.isfile(abs_path):
        abs_path = rel_path

    body = load_pybullet(abs_path, **kwargs)
    if pose is not None:
        set_pose(body, pose)
    return body

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
