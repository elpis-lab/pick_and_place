from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            output='screen',
            parameters=['config/ur10_controllers.yaml'],  # Path to your controller config
            remappings=[
                ('/joint_states', '/joint_states')  # Adjust this based on your joint states topic
            ]
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_trajectory_controller'],
            output='screen',
        ),
    ])
