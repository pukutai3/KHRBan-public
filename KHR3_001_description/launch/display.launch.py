from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import xacro


def generate_launch_description():
    package_share = get_package_share_directory('KHR3_001_description')
    model_path = os.path.join(package_share, 'urdf', 'KHR3_001.xacro')
    rviz_path = os.path.join(package_share, 'launch', 'urdf.rviz')
    robot_description = xacro.process_file(model_path).toxml()

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
            output='screen',
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            output='screen',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_path],
            output='screen',
        ),
    ])
