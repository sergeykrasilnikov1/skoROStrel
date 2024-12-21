import os

from ament_index_python import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.substitutions import TextSubstitution
from launch_ros.actions import Node
from launch_ros.actions import SetParameter


def generate_launch_description():
    
    
    config_comp = os.path.join(
      get_package_share_directory('autorace_camera'),
      'calibration',
      'extrinsic_calibration',
      'compensation.yaml'
      )
    
    config_proj = os.path.join(
      get_package_share_directory('autorace_camera'),
      'calibration',
      'extrinsic_calibration',
      'projection.yaml'
      )
    
    node_compensation = Node(
            package='autorace_camera',
            namespace='camera',
            executable='image_compensation',
            name='image_compensation',
            parameters=[config_comp],
      )
    
    node_projection = Node(
            package='autorace_camera',
            namespace='camera',
            executable='image_projection',
            name='image_projection',
            parameters=[config_proj]
 
      )
    
    node_compensation_projection = Node(
            package='autorace_camera',
            namespace='camera',
            executable='image_compensation',
            name='image_compensation_projection',
            parameters=[config_comp],
      )
    
    ld = LaunchDescription([node_compensation,
                          node_projection,
                          node_compensation_projection,
                          SetParameter(name='use_sim_time', value=True)])
                          # 'use_sim_time' will be set on all nodes following the line above
    
    return ld
