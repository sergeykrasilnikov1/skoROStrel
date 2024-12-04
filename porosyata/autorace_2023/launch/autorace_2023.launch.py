import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('robot_bringup'), 'launch'),
                                       '/autorace_2023.launch.py']),
    )

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('autorace_camera'), 'launch'),
                                        "/extrinsic_camera_calibration.launch.py"]),
    )


    return LaunchDescription([
        bringup,
        camera,
        Node(
            package="autorace_2023",
            executable="detect",
            name="sign_detection",
        ),
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'base_frame_id': 'base_link',
                'odom_frame_id': 'odom',
                'scan_topic': 'scan'
            }],
        ),
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[{
                'use_sim_time': True,  # Set to True if using simulation
                'default_bt_xml_filename': 'navigate_w_replanning_and_recovery.xml',
            }],
        ),
        # Node(
        #     package="autorace_2023",
        #     executable="control",
        #     name="control"
        # ),
        # Node(
        #     package="autorace_2023",
        #     executable="special",
        #     name="hard_code"
        # ),
        Node(
            package='autorace_2023',
            executable='detect_lane',
            name='lane_detection',
        ),
        # Node(
        #     package="autorace_2023",
        #     executable="pid_lane",
        #     name="PID",
        # ),

        # Node(
        #     package="autorace_2023",
        #     executable="pid_pixels",
        #     name="pid_pixels",
        # ),   

        Node(
            package="referee_console",
            executable="mission_autorace_2023_referee",
            name="referee"
        ),

    ])
