import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import os
from std_msgs.msg import String
from ultralytics import YOLO
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from nav_msgs.msg import Odometry
import time
from nav_msgs.msg import Odometry
from math import radians, degrees, atan2, sqrt
import math

class TrafficSignDetector(Node):
    def __init__(self):
        super().__init__('traffic_sign_detector')

        self.sub_image = self.create_subscription(Image, '/color/image', self.image_callback, 1)
        self.pub_image = self.create_publisher(Image, '/traffic_sign_detector/image', 1)
        self.sub_lidar = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 1)

        self.cvBridge = CvBridge()
        
        self.model = YOLO(os.getcwd() + '/src/treponema/autorace_2023/images/best.pt')
        self.is_active = False
        self.control_state_pub = self.create_publisher(String, '/control/active_node', 1)
        self.control_state_sub = self.create_subscription(String, '/control/active_node', self.control_state_callback, 1)

        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        

        self.finish = self.create_publisher(String, '/robot_finish', 1)

        self.lidar_active = False
        self.signs_activated = [""]
        self.timer_group = MutuallyExclusiveCallbackGroup()
        self.timer = None

        # self.wall_follow_distance = 0.1
        # self.front_threshold = 0.38
        # self.state = 'follow_left_wall'
        # self.hysteresis = 0.01
        # self.last_state_change_time = 0
        self.parking_active = False

        self.tonnel_state = 0
        # self.turning_complete = False
        self.leave_tonnel = False

        self.target_angle = 0
        self.angle_tolerance = 1.0  # Допустимая ошибка в градусах
        
        # Внутренние переменные
        self.current_yaw = 0.0
        self.initial_yaw = None


        self.angle_to_turn = 0.0

        self.target_distance = 0.0  # Заданное расстояние (в метрах)
        self.linear_speed = 0.2  # Скорость движения вперед (м/с)
        self.distance_tolerance = 0.01  # Допуск ошибки расстояния (м)

        # Внутренние переменные
        self.start_x = None
        self.start_y = None
        self.current_x = None
        self.current_y = None


        self.min_rad = None
        self.max_rad = None

        self.position = None
        self.orientation = None

        self.linear_velocity = None
        self.angular_velocity = None

        self.start_tonnel = False

        self.current_task = None



        self.command_queue = []  # Очередь команд
        self.executing_command = False  # Флаг выполнения команды

        self.timer2 = self.create_timer(0.1, self.execute_commands)

        self.is_parking = False
        self.parking_start = False
        self.is_repair_work = False

    def add_rotation(self, angle_deg):
        """Добавляем команду на поворот в очередь"""
        self.command_queue.append(('rotate', angle_deg))
        self.get_logger().info(f"Added rotation command: {angle_deg} degrees.")

    def add_drive(self, distance_m):
        """Добавляем команду на движение в очередь"""
        self.command_queue.append(('drive', distance_m))
        self.get_logger().info(f"Added drive command: {distance_m} meters.")

    def execute_commands(self):
        if self.executing_command or not self.command_queue:
            return  # Если команда выполняется или очередь пуста, ничего не делаем

        # Извлекаем следующую команду из очереди
        command_type, parameter = self.command_queue.pop(0)

        if command_type == 'rotate':
            self.start_rotation(parameter)
        elif command_type == 'drive':
            self.start_drive(parameter)


    def start_rotation(self, angle_deg):
        """Начинаем выполнение команды поворота"""
        self.target_angle = angle_deg
        self.initial_yaw = None
        self.executing_command = True
        self.get_logger().info(f"Starting rotation to {angle_deg} degrees.")
        self.timer2 = self.create_timer(0.01, self.rotate_callback)
        


    def process_mode(self):
        if self.get_odometry() is None:
            time.sleep(1)

        # Проверка выполнения задания
        if self.current_task is not None:
            if not self.is_task_completed():
                return None

    def filter_lidar_angles(self, distances, odom, robot=None):
        # new_distances = []
        sector_step = 20
        sectors = 18
        max_i = -1
        max_dst = -1

        orient = odom['orient']
        left_range = self.max_rad - orient      # на сколько радиан слева можем повернуть
        right_range = orient - self.min_rad     # на сколько радиан справа можем повернуть

        max_left_index = math.ceil(np.degrees(left_range) / sector_step)     # сколько индексов слева можем рассматривать
        max_right_index = math.ceil(np.degrees(right_range) / sector_step)   # сколько индексов слева можем рассматривать
        for i, distance in enumerate(distances):
            if i > max_left_index and i < (sectors - max_right_index):
                continue
            if distance > max_dst and distance < 10:
                max_dst = distance
                max_i = i
        if max_i > (sectors / 2):
            self.angle_diff = -np.radians((sectors - max_i) * sector_step)
        else:
            self.angle_diff = np.radians(max_i * sector_step)
        self.distance_diff = max_dst * 0.6

    def get_sectored_lidar(self):
        try:
            normalized_distances = np.nan_to_num(
                self.distances, nan=0, posinf=10, neginf=0)

            sector_step = 20
            window_radius = 10
            # Рассчитываем усредненные значения для каждого сектора
            sectored_distances = []
            for center_index in range(0, 360, sector_step):
                # Определяем границы окна
                start_index = (center_index - window_radius) % 360
                end_index = (center_index + window_radius + 1) % 360
                # Собираем подмассив
                if start_index < end_index:
                    window = normalized_distances[start_index:end_index]
                else:
                    window = np.concatenate(
                        (normalized_distances[start_index:], normalized_distances[:end_index]))
                # Берем среднее из трех наименьших значений
                # mean_of_smallest = np.mean(np.partition(window, 2)[:3])
                # sectored_distances.append(mean_of_smallest)
                # Берем наименьшее
                sectored_distances.append(min(window))
            return sectored_distances
        except Exception as e:
            log(self, F"Error in robot.get_sectored_lidar: {e}", 'ERROR')


    def move(self, linear_x=0.0, angular_z=0.0):
        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.angular.z = angular_z
        self.pub_cmd_vel.publish(cmd)

    def move_task(self, distance, linear_x=0.35):
        odom = self.get_normalized_odometry()
        new_x = odom['pos'][0] + np.cos(odom['orient']) * distance
        new_y = odom['pos'][1] + np.sin(odom['orient']) * distance
        self.target_odom = {
            'pos': (new_x, new_y),
            'orient': odom['orient'],
            'linear_v': linear_x,
            'angular_v': odom['angular_v']
        }
        self.move(linear_x=linear_x)
        self.current_task = 'move'
        return 0

    def rotate_task(self, angle, angular_v=np.pi / 2):
        odom = self.get_normalized_odometry()
        new_orient = odom['orient'] + angle
        self.target_odom = {
            'pos': odom['pos'],
            'orient': new_orient,
            'linear_v': odom['linear_v'],
            'angular_v': angular_v
        }
        self.move(angular_z=angular_v)
        self.current_task = 'rotate'
        return 0

    def is_task_completed(self, epsilon=0.1, min_v=0.025, max_v=0.75, min_w=np.pi / 8, max_w=np.pi / 3, safe_mode=True):
        """Проверяет выполнено ли задание, возвращает ответ, регулирует скорость.
            Должна первоочередно вызываться в robot.process_mode() для правильной обработки заданий.

        Args:
            epsilon (float, optional): Точность для линейных заданий, epsilon / 8 для вращательных заданий.
            min_v (float, optional):
            max_v (float, optional): Стандартная линейная скорость, которая умножается на величину ошибки между целевыми координатами и текущими.
            min_w (_type_, optional):
            max_w (_type_, optional): Стандартная скорость вращения, которая умножается на величину ошибки между целевым радианом и текущим.

        Returns:
            _type_: True если задание выполнены, иначе False
        """
        odom = self.get_normalized_odometry()

        err_x = max(self.target_odom['pos'][0], odom['pos'][0]) - min(self.target_odom['pos'][0], odom['pos'][0])
        err_y = max(self.target_odom['pos'][1], odom['pos'][1]) - min(self.target_odom['pos'][1], odom['pos'][1])
        err_a = self.target_odom['orient'] - odom['orient']
        if safe_mode is True:
            distances = self.get_sectored_lidar()

        if self.current_task == 'move':
            if safe_mode:
                if min(distances[-1], distances[0], distances[1]) < 0.2:
                    self.current_task = None
                    self.target_odom = None
                    self.move(-0.6, 0.3)
                    time.sleep(0.01)
                    return True
            if abs(err_x) < epsilon and abs(err_y) < epsilon:
                self.current_task = None
                self.target_odom = None
                self.move(0.0, 0.0)
                time.sleep(0.01)
                return True
            new_v = max(err_x, err_y) * max_v
            self.move(linear_x=new_v)
            return False
        elif self.current_task == 'rotate':

            if abs(err_a) < 0.1:
                self.current_task = None
                self.target_odom = None
                self.move(0.0, 0.0)
                time.sleep(0.01)
                return True
            new_w = err_a * max_w
            if new_w > max_w:
                new_w = max_w
            elif new_w < -max_w:
                new_w = -max_w

            self.move(angular_z=new_w)
            return False
        return False

    def normalize_angle(self, angle):
        """Нормализуем угол в диапазон [-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


    def get_odometry(self):
        return {
            'position': self.position,
            'orientation': self.orientation,
            'linear_velocity': self.linear_velocity,
            'angular_velocity': self.angular_velocity,
        }

    def get_normalized_odometry(self):
        try:
            odom = self.get_odometry()

            # Переводим блядские кватернионы в радианы
            def quaternion_to_z_angle(q):
                # q is a quaternion [x, y, z, w]
                w = q[3]
                theta = 2 * np.arccos(w)
                return theta
            q = [
                    odom['orientation'].x,
                    odom['orientation'].y,
                    odom['orientation'].z,
                    odom['orientation'].w
            ]
            z = quaternion_to_z_angle(q)

            normalized_odom = {
                'pos': (odom['position'].x, odom['position'].y),
                'orient': self.normalize_angle(z),
                'linear_v': odom['linear_velocity'].x,
                'angular_v': odom['angular_velocity'].z
            }
            return normalized_odom
        except Exception as e:
            time.sleep(1)
            self.get_logger().info(f"error odom {e}")
            self.get_normalized_odometry()


    def odom_callback(self, msg):
        # Получаем текущий угол из одометрии
        orientation_q = msg.pose.pose.orientation
        _, _, yaw = self.euler_from_quaternion(orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)
        self.current_yaw = degrees(yaw)  # Конвертируем в градусы

        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        self.position = msg.pose.pose.position
        self.orientation = msg.pose.pose.orientation
        self.linear_velocity = msg.twist.twist.linear
        self.angular_velocity = msg.twist.twist.angular


    def calculate_distance(self, x1, y1, x2, y2):
        # Вычисляем евклидово расстояние между двумя точками
        return sqrt((x2 - x1)**2 + (y2 - y1)**2)


    def drive_callback(self):
        if self.target_distance is None:
            return

        if self.start_x is None or self.start_y is None:
            # Запоминаем начальную позицию робота
            self.start_x = self.current_x
            self.start_y = self.current_y
            # self.get_logger().info(f"Starting position: x={self.start_x:.2f}, y={self.start_y:.2f}")
            return

        # Вычисляем пройденное расстояние
        distance_travelled = self.calculate_distance(self.start_x, self.start_y, self.current_x, self.current_y)
        self.get_logger().info(f"Distance travelled: {distance_travelled:.2f} m")

        # Проверяем, достиг ли робот целевого расстояния
        if distance_travelled >= self.target_distance - self.distance_tolerance:
            self.stop_robot()
            self.get_logger().info("Target distance reached!")
            self.executing_command = False
            self.timer2.cancel()
            return

        # Публикуем линейную скорость для движения вперед
        twist_msg = Twist()
        twist_msg.linear.x = (self.target_distance - distance_travelled) * 0.8 + 0.03
        self.pub_cmd_vel.publish(twist_msg)
    
    def euler_from_quaternion(self, x, y, z, w):
        # Преобразование кватерниона в углы Эйлера (roll, pitch, yaw)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)

        return roll, pitch, yaw



    def start_drive(self, distance_m):
        self.target_distance = distance_m
        self.start_x = self.current_x
        self.start_y = self.current_y
        self.executing_command = True
        self.get_logger().info(f"Starting drive for {distance_m} meters.")
        self.timer2 = self.create_timer(0.01, self.drive_callback)

    def rotate_callback(self):
        if self.target_angle is None:
            return
        
        if self.initial_yaw is None:
            self.initial_yaw = self.current_yaw
            self.target_yaw = (self.initial_yaw + self.target_angle) % 360
            self.get_logger().info(f"Starting rotation: Target Yaw = {self.target_yaw:.2f}")
        
        # Рассчитываем ошибку угла
        angle_diff = (self.target_yaw - self.current_yaw + 360) % 360
        if angle_diff > 180:  # Двигаемся в минимальном направлении
            angle_diff -= 360

        self.get_logger().info(f"Angle Diff: {angle_diff:.2f}")

        # Проверяем, достигли ли цели
        if abs(angle_diff) <= self.angle_tolerance:
            self.stop_robot()
            self.get_logger().info("Target angle reached!")
            self.executing_command = False
            self.timer2.cancel()
            return
        
        # Публикуем угловую скорость
        twist_msg = Twist()
        twist_msg.angular.z = np.sign(angle_diff)*((angle_diff/40)**2 + 0.1) # Поворачиваем в нужную сторону
        self.pub_cmd_vel.publish(twist_msg)
    
    def stop_robot(self):
        # Останавливаем робота
        stop_msg = Twist()
        self.pub_cmd_vel.publish(stop_msg)
        self.get_logger().info("Robot stopped.")

    def control_state_callback(self, msg):
        msg, turn_speed, speed = msg.data.split()
        if msg == 'parking' and 'parking_flag' not in self.signs_activated:
            self.lidar_active = True
            self.parking_active = True
            self.signs_activated.append('parking_flag')
        else:
            self.is_active = (msg == 'traffic_sign_detector')

    def set_active_node(self, node_name):
        self.is_active = not self.is_active
        msg = String()
        msg.data = node_name
        self.control_state_pub.publish(msg)

    def image_callback(self, msg_img):
        cv_image_original = self.cvBridge.imgmsg_to_cv2(msg_img, "bgr8")

        results = self.model(cv_image_original)
        detected_sign = self.process_yolo_results(results)

        if detected_sign:
            self.execute_instruction(detected_sign)

        annotated_image = self.draw_bboxes(cv_image_original, results)
        self.pub_image.publish(self.cvBridge.cv2_to_imgmsg(annotated_image, "bgr8"))
        if self.lidar_active:
            self.check_color(cv_image_original)


    def check_color(self, cv_image_original):
        lower_dark_blue = np.array([100, 100, 50])
        upper_dark_blue = np.array([120, 255, 200])
        hsv = cv2.cvtColor(cv_image_original, cv2.COLOR_BGR2HSV)
        mask_dark_blue = cv2.inRange(hsv, lower_dark_blue, upper_dark_blue)

        dark_blue_pixels = cv2.countNonZero(mask_dark_blue)
        if dark_blue_pixels > 3500:
            self.execute_instruction('parking')

        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        green_pixels = cv2.countNonZero(mask_green)
        twist = Twist()
        # self.get_logger().info(f"green_pixels {green_pixels}")
        if 100 < green_pixels < 2000 and not self.leave_tonnel and 'tonnel' in self.signs_activated:
            self.leave_tonnel = True
            self.current_task = None
            self.set_active_node('traffic_sign_detector 100 0.25')
        #     self.set_active_node('lane_follower 100 0.1')
            twist.linear.x = 0.0 
            twist.angular.z = -0.5  
            self.pub_cmd_vel.publish(twist)
            time.sleep(2)
            # twist.linear.x = 0.2 
            # twist.angular.z = 0.00  
            # self.pub_cmd_vel.publish(twist)
        if 3500 < green_pixels and 'tonnel' in self.signs_activated:  
            self.set_active_node('traffic_sign_detector 100 0.25')
            text = String()
            text.data = 'treponema'
            self.finish.publish(text)
            time.sleep(0.2)
            
            twist.linear.x = 0.0 
            twist.angular.z = 0.0  
            self.pub_cmd_vel.publish(twist)
            time.sleep(99999)

        lower_yellow = np.array([20, 100, 230])
        upper_yellow = np.array([30, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        

    def change_state(self, new_state, delay=7, tonnel=False):
        """Меняет состояние с проверкой на минимальный интервал времени."""
        current_time = time.time()
        if current_time - self.last_state_change_time < delay: 
            # self.get_logger().info(f"Cannot change state to {new_state}. Waiting for cooldown.")
            return

        # self.get_logger().info(f"Changing state to {new_state}")
        if not tonnel:
            self.state = new_state
        else:
            self.tonnel_state = new_state
        self.last_state_change_time = current_time


    def lidar_callback(self, msg):
        if not self.lidar_active:
            return

        ranges = np.array(msg.ranges)
        self.distances = msg.ranges
        ranges = np.where(np.isfinite(ranges), ranges, 20)  # Обработка NaN/Inf

        front_range = ranges[:25].tolist() + ranges[-25:].tolist()
        left_range = ranges[60:120]
        right_range = ranges[-120:-60]


        min_front_dist = min(front_range)
        avg_left_dist = np.mean(left_range)
        avg_right_dist = np.mean(right_range)


        twist = Twist()
        if 'tonnel' in self.signs_activated:
            self.handle_tonnel(twist, front_range, avg_left_dist, avg_right_dist, np.array(msg.ranges), msg)
        elif 'cross_walk' in self.signs_activated:
            self.handle_crosswalk(twist, ranges)
        elif self.parking_active:
            self.get_logger().info(f"self.command_queue: {self.command_queue}")
            if not self.parking_start:
                self.parking_start = True
                self.add_rotation(-5)
                self.add_drive(0.22)

                
            elif self.parking_start and not self.command_queue:
                self.get_logger().info(f"self.command_queue: {self.command_queue}")
                self.handle_parking(twist, left_range, right_range)
        else:
            if not self.executing_command:
                ranges = np.array(msg.ranges)
                ranges = np.where(np.isfinite(ranges), ranges, 0)  # Обработка NaN/Inf

                left_range = ranges[60:120]
                right_range = ranges[-120:-60]

                avg_left_dist = np.mean(left_range)
                avg_right_dist = np.mean(right_range)
                self.follow_wall(twist, avg_left_dist, avg_right_dist, min_front_dist)
                

    def follow_wall(self, twist, avg_left_dist, avg_right_dist, min_front_dist):
        # self.get_logger().info(f"Lef: {avg_left_dist}, Right range: {avg_right_dist} ")
        if min_front_dist<0.5 and not self.is_repair_work:
            self.set_active_node('traffic_sign_detector 100 0.25')
            twist = Twist()
            twist.linear.x = 0.1
            twist.angular.z = 0.0
            self.pub_cmd_vel.publish(twist)
        if not self.is_repair_work and min_front_dist<0.25:
            
            self.is_repair_work = True
            self.add_rotation(88)
            self.add_drive(0.32)
            self.add_rotation(-90)
            self.add_drive(0.45)
            self.add_rotation(-80)
            self.add_drive(0.35)
            self.add_rotation(95)
            self.add_drive(0.4)
            self.add_rotation(75)
            self.add_drive(0.10)
        if  not self.command_queue and self.is_repair_work == True:
            self.target_angle = None
            self.lidar_active = False
            self.set_active_node('lane_follower_yellow 100 0.25')


    # def follow_wall(self, twist, avg_left_dist, avg_right_dist, min_front_dist):

    # # Следование за левой стеной или правой
    #     self.get_logger().info(f":self.state {self.state} avg_left_dist {avg_left_dist}")
    #     # if self.avoid_obstacles:
    #     self.turn_speed = 1.1
    #     self.forward_speed = 0.157
    #     if self.state == 'follow_left_wall':
    #         if min_front_dist < self.front_threshold:
    #             self.change_state('follow_right_wall') 
    #             twist.linear.x = 0.0
    #             twist.angular.z = -self.turn_speed * 0.5
    #         elif avg_left_dist > self.wall_follow_distance + self.hysteresis:
    #             twist.linear.x = self.forward_speed * 0.8
    #             twist.angular.z = self.turn_speed * 0.5
    #         elif avg_left_dist < self.wall_follow_distance - self.hysteresis:
    #             twist.linear.x = self.forward_speed
    #             twist.angular.z = -self.turn_speed
    #         else:
    #             twist.linear.x = self.forward_speed
    #             twist.angular.z = 0.0

    #     elif self.state == 'follow_right_wall':
    #         if min_front_dist < self.front_threshold:
    #             self.change_state('follow_left_wall') 
    #             twist.linear.x = 0.05
    #             twist.angular.z = self.turn_speed * 0.5
    #         elif avg_right_dist > self.wall_follow_distance + self.hysteresis:
    #             twist.linear.x = self.forward_speed * 0.8
    #             twist.angular.z = -self.turn_speed * 0.5
    #         elif avg_right_dist < self.wall_follow_distance - self.hysteresis:
    #             twist.linear.x = self.forward_speed
    #             twist.angular.z = self.turn_speed
    #         else:
    #             twist.linear.x = self.forward_speed * 0.5
    #             twist.angular.z = 0.0
    #     self.pub_cmd_vel.publish(twist)


    def handle_parking(self, twist, left_dist, right_dist):

        if self.is_parking == False:
            self.is_parking = True
            self.get_logger().info("Parking mode activated")

            self.get_logger().info(f"Left range: {left_dist}, Right range: {right_dist}")

            if min(right_dist) < 0.4: 
                park_side = 'left'
                self.get_logger().info("Parking on the left side.")
            else:  # Машина справа
                
                park_side = 'right'
                self.get_logger().info("Parking on the right side.")
            self.get_logger().info(f" command_queue {self.command_queue}")
            if park_side == 'right':
                self.add_rotation(-10)
                self.add_drive(0.14)
                self.add_rotation(-80)
                self.add_drive(0.24)
                self.add_rotation(-90)
                self.add_rotation(-30)
                self.add_drive(0.39)
                self.add_rotation(40)
                self.add_drive(0.1)
            else:  # park_side == 'left'
                self.add_rotation(10)
                self.add_drive(0.13)
                self.add_rotation(80)
                self.add_drive(0.24)
                self.add_rotation(90)
                self.add_rotation(35)
                self.add_drive(0.40)
                self.add_rotation(-40)
                self.add_drive(0.1)
        if  not self.command_queue and self.is_parking == True:
            self.parking_active = False
            self.lidar_active = False
            self.set_active_node('lane_follower_yellow 100 0.25')
            self.get_logger().info("Parking completed")
            time.sleep(12)
            self.set_active_node('lane_follower 125 0.25')


    def handle_crosswalk(self, twist, ranges):
        front_range = ranges[:18].tolist() + ranges[-18:].tolist()
        min_front_dist = min(front_range)
        self.get_logger().info(f"Checking for obstacles at crosswalk... {min_front_dist}")
        if min_front_dist < 0.6:  
            self.get_logger().info("Obstacle detected at crosswalk. Stopping.")
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.pub_cmd_vel.publish(twist)
            return
        else:
            self.get_logger().info("No obstacle detected. Speeding up to cross quickly.")
            twist.linear.x = 0.3
            twist.angular.z = 0.0
            self.pub_cmd_vel.publish(twist)
            time.sleep(2) 

        self.lidar_active = False 
        self.set_active_node('lane_follower 100 0.25') 


    def handle_tonnel(self, twist, front_range, avg_left_dist, avg_right_dist, ranges, msg):
        if avg_right_dist > 0.4 and self.tonnel_state ==0:
            twist.linear.x = 0.0  
            twist.angular.z = -0.4
            self.pub_cmd_vel.publish(twist)
            return
        if not self.leave_tonnel:
            

            sectored_distances = self.get_sectored_lidar()
            if min(ranges[60:120]) < 0.05:
                twist.linear.x = 0.2
                twist.angular.z = -0.3
                self.pub_cmd_vel.publish(twist)
                time.sleep(0.1)
                return
            # if min(front_range) <0.1:
            #         # robot.move(linear_x=0.0, angular_z=0.0):
            #     twist.linear.x = -0.2
            #     twist.angular.z = 0.0
            #     self.pub_cmd_vel.publish(twist)
            #     time.sleep(0.4)
            #     return
            odom = self.get_normalized_odometry()

            if self.tonnel_state ==0:
                    
                self.min_rad = odom['orient']
                self.max_rad = odom['orient'] + np.pi / 2
                self.get_logger().info(f"odom: {odom} sectored_distances {sectored_distances}")
                self.tonnel_state+=1

            # Step 1: Determine the angle to turn
            elif self.tonnel_state ==1 and not self.current_task:
            # elif self.tonnel_state ==1:
                
                self.filter_lidar_angles(sectored_distances, odom)
                # if self.angle_diff!=0:
                self.rotate_task(self.angle_diff)
                    # time.sleep(0.1)
                self.tonnel_state+=1
                self.get_logger().info(f"Target angle: {self.angle_diff:.2f} degrees, Target distance: {self.distance_diff:.2f} m self.tonnel_state {self.tonnel_state}")
            # Step 3: Move forward after completing the turn
            else:
                if not self.current_task:
                    self.move_task(self.distance_diff * 0.5)
                    self.tonnel_state-=1
        else:
            self.get_logger().info(f'min(front_range) {min(front_range)}')
            if avg_right_dist > 3 and avg_left_dist > 3:
                self.set_active_node('lane_follower 75 0.25')
            if min(front_range) > 0.2:
                twist.linear.x = 0.08 
                twist.angular.z = 0.0 
            else:
                twist.linear.x = 0.0  
                twist.angular.z = 0.4 
            self.pub_cmd_vel.publish(twist)

        

    # def handle_tonnel(self, twist, front_range, avg_left_dist, avg_right_dist, ranges):
    #     min_front_dist = min(ranges[:18].tolist() + ranges[-18:].tolist())
    #     if not self.leave_tonnel:
    #         avg_front_dist = np.mean(front_range)
    #         n = 10 
    #         edges = front_range[:n] + front_range[-n:]
    #         center = front_range[n:-n]
    #         mean_edges = np.mean(edges)
    #         mean_center = np.mean(center)

    #         difference = mean_center - mean_edges
    #         # self.get_logger().info(f"self.tonnel_state {min_front_dist}")

            

    #         if min_front_dist < 0.3:  # Препятствие впереди
        
    #             if self.tonnel_state == 'left':
    #                 if avg_left_dist < 0.5:
    #                     twist.linear.x = 0.01  
    #                     twist.angular.z = -0.7 
    #                     self.turning_complete = False 
    #                 elif not self.turning_complete:
    #                     twist.linear.x = 0.01 
    #                     twist.angular.z = 0.7  
    #                 else:
    #                     twist.linear.x = 0.3 
    #                     twist.angular.z = 0.0  
    #                     self.change_state("right", 5, True)
    #                     self.turning_complete = False  
    #             else:
    #                 if not self.turning_complete:
    #                     twist.linear.x = 0.01
    #                     twist.angular.z = -0.7 
    #                 else:
    #                     twist.linear.x = 0.3  
    #                     twist.angular.z = 0.0 
    #                     self.change_state("left", 5, True)
    #                     self.turning_complete = False  
    #         else: 
    #             twist.linear.x = 0.2 
    #             twist.angular.z = 0.0

    #             self.turning_complete = True
    #         self.pub_cmd_vel.publish(twist)
        # else:
        #     self.get_logger().info(f"leave_mod {True}")
        #     if min_front_dist < 0.25 or avg_right_dist < 0.1:
        #         twist.linear.x = 0.1 
        #         twist.angular.z = 0.5  
        #     elif min_front_dist > 1 and avg_right_dist > 2:
        #         self.lidar_active = False
                
        #         self.set_active_node('lane_follower 75')
        #         time.sleep(8)
        #         self.set_active_node('traffic_sign_detector 100')
        #         msg = String()
        #         msg.data = 'finish'
        #         self.finish.publish(msg)
        #         twist.linear.x = 0.0 
        #         twist.angular.z = 0.0  
        #         self.pub_cmd_vel.publish(twist)
        #     else:
        #         twist.linear.x = 0.25 
        #         twist.angular.z = -0.1  

        # self.pub_cmd_vel.publish(twist)


    def process_yolo_results(self, results):
        for result in results:
            boxes = result.boxes 
            for box in boxes:
                class_id = int(box.cls) 
                confidence = box.conf[0] 
                label = result.names[class_id]  
                xyxy = box.xyxy[0].cpu().numpy().astype(int)  
                x1, y1, x2, y2 = xyxy
                area = (x2 - x1) * (y2 - y1)
                self.get_logger().info(f"Box area: {area}, Label: {label}, Confidence: {confidence}")
                if confidence > 0.8 and area > 25000 or label=='tonnel' and confidence > 0.8:
                    return label
        return None

    def draw_bboxes(self, image, results):
        for result in results:
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)  
                class_id = int(box.cls)
                label = result.names[class_id]
                confidence = box.conf[0]

                cv2.rectangle(image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                cv2.putText(image, f"{label}: {confidence:.2f}", (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return image

    def execute_instruction(self, sign_name):
        if sign_name in self.signs_activated:
            return
        self.signs_activated.append(sign_name)
        if not self.is_active and sign_name not in ['tonnel']: 
            self.set_active_node('traffic_sign_detector 100 0.25')

        twist = Twist()
        time.sleep(0.1)
        if sign_name == 'turn':
            self.set_active_node('lane_follower 100 0.18')
            return
        elif sign_name == "turn_left":
            self.get_logger().info("Executing turn left")

            twist.linear.x = 0.0
            twist.angular.z = 1.9
            self.pub_cmd_vel.publish(twist)
            time.sleep(1.1)
            # self.add_rotation(40)
            self.set_active_node('lane_follower 100 0.4')
        elif sign_name == "turn_right":
            self.get_logger().info("Executing turn right")
            twist.linear.x = 0.0
            twist.angular.z = -2.0
            self.pub_cmd_vel.publish(twist)
            time.sleep(1.2)
            # self.add_rotation(-40)
            self.set_active_node('lane_follower 100 0.4')
        elif sign_name == "repair_work":
            self.get_logger().info("Executing repair_work")
            # time.sleep(1.0)
            # self.set_active_node('traffic_sign_detector 100 0.25')
            # self.add_rotation(-5)
            # self.add_drive(0.72)
            # twist.linear.x = 0.28
            # twist.angular.z = 0.0
            # self.pub_cmd_vel.publish(twist)
            # time.sleep(3.0)
            # twist.linear.x = 0.0
            # twist.angular.z = 0.0
            # self.pub_cmd_vel.publish(twist)
            # time.sleep(1.0)
            self.set_active_node('lane_follower_white 100 0.25')
            # return
            # time.sleep(1.0)
            # twist.linear.x = 0.1
            # twist.angular.z = 0.0
            # self.pub_cmd_vel.publish(twist)
            # time.sleep(1.3)
            self.lidar_active = True
        elif sign_name == "parking":
            self.lidar_active = False
            twist.linear.x = 0.2
            twist.angular.z = 0.5
            self.pub_cmd_vel.publish(twist)
            time.sleep(2.2)
            self.parking_active = False  # Активируем парковку
            self.set_active_node('lane_follower_yellow 100 0.25')
            return
        elif sign_name == "cross_walk":
            self.lidar_active = True
            self.set_active_node('traffic_sign_detector 100 0.25')
        elif sign_name == "tonnel":
            self.create_timer(0.2, self.process_mode)
            time.sleep(1.1)
            self.set_active_node('traffic_sign_detector 100 0.25')
            twist.linear.x = 0.2
            twist.angular.z = 0.05
            self.pub_cmd_vel.publish(twist)
            time.sleep(4.5)
            
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.pub_cmd_vel.publish(twist)
            
            time.sleep(2.0)
            self.lidar_active = True  
            

        if not self.lidar_active:    
            self.set_active_node('lane_follower 75 0.25')
            self.get_logger().info(f"detect active {self.is_active}")


def main(args=None):
    rclpy.init(args=args)
    node = TrafficSignDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
