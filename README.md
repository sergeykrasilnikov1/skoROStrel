# Treponema

autorace 2024
```
cd ros2_ws/src
```
```
git clone https://github.com/sergeykrasilnikov1/skoROStrel
```
```
cd ..
```
```
pip install -r requirements.txt
```
```
colcon build --packages-select autorace_2023 autorace_camera autorace_msgs
```
```
ros2 launch autorace_2023 autorace_2023.launch.py
```
