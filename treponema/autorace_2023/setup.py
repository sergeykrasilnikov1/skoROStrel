from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'autorace_2023'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'images', 'templates'), glob(os.path.join('images', 'templates', '*.png'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='treponema',
    maintainer_email="s.krasiknikov2@g.nsu.ru",
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "detect = autorace_2023.detect:main",
            "detect_lane = autorace_2023.detect_lane:main",
        ],
    },
)
