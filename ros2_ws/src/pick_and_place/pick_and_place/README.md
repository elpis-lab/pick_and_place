This file explains how to run Edge Grasp with real pointcloud.

1. Run the following command to turn on camera:  ros2 launch realsense2_camera rs_launch.py depth_module.depth_profile:=640x480x30
2. Run the following command to run ROS node that takes the depth map at the instance it is run to pass it to edge grasp network: ros2 run pick_and_place depth_sub
3. This node calls the gen_grasp funciton in edge_grasp.py which is located in the /home/sultan/Edge-Grasp-Network
