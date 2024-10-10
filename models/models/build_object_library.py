"""Build URDF for objects. Scripts from https://github.com/harvard-microrobotics/object2urdf"""

from object_urdf import ObjectUrdfBuilder

# Build entire libraries of URDFs
# This is only suitable for objects built with single obj/stl file
# Models such as robots or articulated objects will not work properly

object_folder = "ycb"
builder = ObjectUrdfBuilder(object_folder)
builder.build_library(
    force_overwrite=True,
    decompose_concave=True,
    force_decompose=False,
    center="mass",
)
