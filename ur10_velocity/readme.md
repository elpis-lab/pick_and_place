THROWING USING ACTION CLIENT

#ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur10 robot_ip:=192.168.1.102 launch_rviz:=false

#ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur10 launch_rviz:=true use_mock_hardware:=true

#ros2 control switch_controllers --activate scaled_joint_trajectory_controller

#ros2 run ur10_velocity ur10_trajectory_sender