import os
import time

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Задержка перед запуском bringup
    bringup = TimerAction(
        period=1.0,  # Задержка в 1 секунду
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory('robot_bringup'), 'launch'),
                    '/autorace_2023.launch.py']),
            )
        ]
    )

    # Задержка перед запуском camera
    camera = TimerAction(
        period=1.0,  # Задержка в 2 секунды
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory('autorace_camera'), 'launch'),
                    "/extrinsic_camera_calibration.launch.py"]),
            )
        ]
    )

    # Задержка перед запуском sign_detection
    sign_detection = TimerAction(
        period=2.0,  # Задержка в 3 секунды
        actions=[
            Node(
                package="autorace_2023",
                executable="detect",
                name="sign_detection",
            )
        ]
    )

    # Задержка перед запуском lane_detection
    lane_detection = TimerAction(
        period=2.0,  # Задержка в 4 секунды
        actions=[
            Node(
                package='autorace_2023',
                executable='detect_lane',
                name='lane_detection',
            )
        ]
    )

    # Задержка перед запуском referee
    referee = TimerAction(
        period=3.0,  # Задержка в 5 секунд
        actions=[
            Node(
                package="referee_console",
                executable="mission_autorace_2023_referee",
                name="referee"
            )
        ]
    )

    

    return LaunchDescription([
        bringup,
        camera,
        sign_detection,
        lane_detection,
        referee,
    ])