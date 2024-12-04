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
from geometry_msgs.msg import PoseStamped
import time
from nav2_simple_commander.robot_navigator import BasicNavigator
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration

class TrafficSignDetector(Node):
    def __init__(self):
        super().__init__('traffic_sign_detector')

        self.sub_image = self.create_subscription(Image, '/color/image', self.image_callback, 1)
        self.pub_image = self.create_publisher(Image, '/traffic_sign_detector/image', 1)
        self.sub_lidar = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 1)

        self.cvBridge = CvBridge()
        
        # Load the YOLO model (path to your model file)
        self.model = YOLO(os.getcwd() + '/src/porosyata/autorace_2023/images/best.pt')  # Update path as needed

        self.is_active = False  # Flag for active status
        self.control_state_pub = self.create_publisher(String, '/control/active_node', 1)
        self.control_state_sub = self.create_subscription(String, '/control/active_node', self.control_state_callback, 1)

        self.lidar_active = False
        self.signs_activated = ['repair_work']
        self.timer_group = MutuallyExclusiveCallbackGroup()
        self.timer = None

        self.forward_speed = 0.14
        self.angular_speed = 1.5
        self.wall_follow_distance = 0.15
        self.front_threshold = 0.35
        self.angle_increment = None
        self.state = 'follow_left_wall'
        self.hysteresis = 0.1
        self.last_state_change_time = 0
        self.parking_active = False

        self.navigator = BasicNavigator()  # Nav2 Basic Navigator
        init_pose = PoseStamped()
        init_pose.header.frame_id = "robot/odom"
        init_pose.header.stamp = self.navigator.get_clock().now().to_msg()
        init_pose.pose.position.x = 0.0
        init_pose.pose.position.y = 0.0
        init_pose.pose.orientation.z = 1.0
        init_pose.pose.orientation.w = 1.0  # Default orientation (facing forward)

        # Set the goal pose for the robot (example goal, modify as needed

        self.navigator.setInitialPose(init_pose)

        self.navigator.lifecycleStartup()  # Ensure Nav2 is ready

    def control_state_callback(self, msg):
        msg, speed = msg.data.split()
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

        # Run YOLO model on the image
        results = self.model(cv_image_original)  # Detect signs in the image

        # Extract the results
        detected_sign = self.process_yolo_results(results)

        # If a sign is detected, execute the corresponding instruction
        if detected_sign:
            self.execute_instruction(detected_sign)

        # Publish the image with bounding boxes
        annotated_image = self.draw_bboxes(cv_image_original, results)
        self.pub_image.publish(self.cvBridge.cv2_to_imgmsg(annotated_image, "bgr8"))

    def change_state(self, new_state):
        """Меняет состояние с проверкой на минимальный интервал времени."""
        current_time = time.time()
        if current_time - self.last_state_change_time < 11:  # Если прошло меньше 2 секунд
            # self.get_logger().info(f"Cannot change state to {new_state}. Waiting for cooldown.")
            return

        # self.get_logger().info(f"Changing state to {new_state}")
        self.state = new_state
        self.last_state_change_time = current_time
    def navigate_out_of_tunnel(self):
        init_pose = PoseStamped()
        init_pose.header.frame_id = "robot/odom"
        init_pose.pose.position.x = 0.0
        init_pose.pose.position.y = 0.0
        init_pose.pose.orientation.w = 1.0  # Default orientation (facing forward)

        # Set the goal pose for the robot (example goal, modify as needed)
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "robot/odom"
        goal_pose.pose.position.x = 2.0  # Example goal X position
        goal_pose.pose.position.y = 2.0  # Example goal Y position
        goal_pose.pose.orientation.w = 1.0  # Default orientation (facing forward)

        self.navigator.setInitialPose(init_pose)


        # Call Nav2 to move to the goal
        path = self.navigator.getPath(init_pose, goal_pose)

        smoothed_path = self.navigator.smoothPath(path)

        # Move the robot towards the goal pose
        self.navigator.goToPose(goal_pose)

        # Monitor navigation feedback and cancel if it takes too long
        while not self.navigator.isTaskComplete():
            feedback = self.navigator.getFeedback()


    def lidar_callback(self, msg):
        if not self.lidar_active:
            return

        # Преобразование данных лидара в массив
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isfinite(ranges), ranges, np.inf)  # Обработка NaN/Inf

        # Разделение данных на зоны
        front_range = ranges[:25].tolist() + ranges[-25:].tolist()
        left_range = ranges[60:120]
        right_range = ranges[-120:-60]

        # Фильтрация некорректных данных
        front_range = [x if np.isfinite(x) else 2 for x in front_range]
        left_range = [x if np.isfinite(x) else 2 for x in left_range]
        right_range = [x if np.isfinite(x) else 2 for x in right_range]

        # Проверка пустоты после фильтрации
        if not front_range or not left_range or not right_range:
            self.get_logger().warning("Lidar data is empty after filtering!")
            return

        # Вычисление расстояний
        min_front_dist = min(front_range)
        avg_left_dist = np.mean(left_range)
        avg_right_dist = np.mean(right_range)
    
        twist = Twist()
        if 'tonnel' in self.signs_activated:
            self.navigate_out_of_tunnel()
            # Проверяем конец тоннеля
            if min_front_dist > 2.0 and avg_left_dist > 2.0 and avg_right_dist > 2.0:
                self.get_logger().info("Tunnel navigation complete. Exiting tunnel.")
                self.tunnel_active = False  # Завершаем режим тоннеля
                self.lidar_active = False
                self.set_active_node('lane_follower 100')  # Возвращаем управление полосой

        elif 'cross_walk' in self.signs_activated:
            front_range = ranges[:15].tolist() + ranges[-15:].tolist()
            front_range = [x if np.isfinite(x) else 2 for x in front_range]
            min_front_dist = min(front_range)
            self.get_logger().info(f"Checking for obstacles at crosswalk... {min_front_dist}")
            if min_front_dist < 0.6:  # Если препятствие ближе 0.5 метра
                self.get_logger().info("Obstacle detected at crosswalk. Stopping.")
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.pub_cmd_vel.publish(twist)
                return
            else:
                self.get_logger().info("No obstacle detected. Speeding up to cross quickly.")
                twist.linear.x = 0.3  # Увеличиваем скорость
                twist.angular.z = 0.0
                self.pub_cmd_vel.publish(twist)
                time.sleep(2)  # Едем быстрее в течение 2 секунд

            self.lidar_active = False  # Отключаем использование лидара для crosswalk
            self.set_active_node('lane_follower 100')  # Возвращаем управление полосой
        elif self.parking_active:
            self.get_logger().info("Parking mode activated")
    
    # Определить расстояния до объекта слева и справа
            self.get_logger().info(f"Left range: {left_range}, Right range: {right_range}")

            # Решаем, с какой стороны парковаться
            if avg_left_dist < avg_right_dist:  # Машина слева, парковаться справа
                park_side = 'right'
                self.get_logger().info("Parking on the right side.")
            else:  # Машина справа, парковаться слева
                park_side = 'left'
                self.get_logger().info("Parking on the left side.")
            
            # twist.linear.x = 0.1
            # twist.angular.z = -1.5
            # self.pub_cmd_vel.publish(twist)
            # time.sleep(0.4)
            # Маневр в зависимости от стороны парковки
            if park_side == 'right':
                twist.linear.x = 0.0
                twist.angular.z = -1.5
                self.pub_cmd_vel.publish(twist)
                time.sleep(0.7)
                twist.linear.x = 0.12
                twist.angular.z = -1.5
                self.pub_cmd_vel.publish(twist)
                time.sleep(4.8)
            else:  # park_side == 'left'
                twist.linear.x = 0.09
                twist.angular.z = 1.5
                self.pub_cmd_vel.publish(twist)
                time.sleep(3.9)

            

            # Финальная корректировка
            twist.linear.x = -1.6
            twist.angular.z = 0.0
            self.pub_cmd_vel.publish(twist)
            time.sleep(1.1)

            # Остановка
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.pub_cmd_vel.publish(twist)

            # Постой 1 секунду
            time.sleep(1)

            # Выезд
            if park_side == 'right':
                twist.linear.x = 0.0
                twist.angular.z = -2.5
                self.pub_cmd_vel.publish(twist)
                time.sleep(1.5)
                twist.linear.x = 0.15
                twist.angular.z = 2.5
                self.pub_cmd_vel.publish(twist)
                time.sleep(2)
            else:  # park_side == 'left'
                twist.linear.x = 0.0
                twist.angular.z = 2.8
                self.pub_cmd_vel.publish(twist)
                time.sleep(1.7)
                twist.linear.x = 0.15
                twist.angular.z = -2.5
                self.pub_cmd_vel.publish(twist)
                time.sleep(3)


            self.parking_active = False
            self.lidar_active = False
            self.set_active_node('lane_follower_yellow 100')
            self.get_logger().info("Parking completed")
            time.sleep(12)
            self.set_active_node('lane_follower 50')
        else:
        # Следование за левой стеной или правой
            if self.state == 'follow_left_wall':
                if min_front_dist < self.front_threshold:
                    self.change_state('follow_right_wall')  # Используем метод смены состояния
                    twist.linear.x = 0.0
                    twist.angular.z = -self.turn_speed * 0.5
                elif avg_left_dist > self.wall_follow_distance + self.hysteresis:
                    twist.linear.x = self.forward_speed
                    twist.angular.z = self.turn_speed * 0.5
                elif avg_left_dist < self.wall_follow_distance - self.hysteresis:
                    twist.linear.x = self.forward_speed
                    twist.angular.z = -self.turn_speed
                else:
                    twist.linear.x = self.forward_speed
                    twist.angular.z = 0.0

            elif self.state == 'follow_right_wall':
                if min_front_dist < self.front_threshold:
                    self.change_state('follow_left_wall')  # Используем метод смены состояния
                    twist.linear.x = 0.05
                    twist.angular.z = self.turn_speed * 0.5
                elif avg_right_dist > self.wall_follow_distance + self.hysteresis:
                    twist.linear.x = self.forward_speed
                    twist.angular.z = -self.turn_speed * 0.5
                elif avg_right_dist < self.wall_follow_distance - self.hysteresis:
                    twist.linear.x = self.forward_speed
                    twist.angular.z = self.turn_speed
                else:
                    twist.linear.x = self.forward_speed * 0.5
                    twist.angular.z = 0.0

            # Публикация команды движения
            self.pub_cmd_vel.publish(twist)

        # Если активирована парковка, выполняем парковку
       

    def process_yolo_results(self, results):
        # Check for detected objects
        for result in results:
            boxes = result.boxes  # Get bounding boxes of detected signs
            for box in boxes:
                class_id = int(box.cls)  # Get the class ID of the detected object
                confidence = box.conf[0]  # Get the confidence score
                label = result.names[class_id]  # Get the label for the detected object
                self.get_logger().info(f"label {label} {confidence}")
                # Check if the detected object is a traffic sign and has a high confidence score
                if confidence > 0.5:
                    self.get_logger().info(f"label {label}")
                    return label
        return None

    def draw_bboxes(self, image, results):
        # Draw bounding boxes on the image for visualization
        for result in results:
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)  # Convert to integer coordinates
                class_id = int(box.cls)
                label = result.names[class_id]
                confidence = box.conf[0]

                # Draw the rectangle
                cv2.rectangle(image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                cv2.putText(image, f"{label}: {confidence:.2f}", (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return image

    def execute_instruction(self, sign_name):
        if sign_name in self.signs_activated:
            return
        self.signs_activated.append(sign_name)
        # rclpy.spin_once(self, timeout_sec=0.2)
        if not self.is_active and sign_name!='tonnel':  # If the node is not active
            self.set_active_node('traffic_sign_detector 100')

        twist = Twist()
        if sign_name == "turn_right":
            self.get_logger().info("Executing turn left")
            # Turn left in an arc
            twist.linear.x = 0.2
            twist.angular.z = 0.0
            self.pub_cmd_vel.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.5)
            for _ in range(15):  # Send commands for turning

                twist.linear.x = 0.1
                twist.angular.z = 1.5
                self.pub_cmd_vel.publish(twist)
                rclpy.spin_once(self, timeout_sec=0.1)
        elif sign_name == "turn_left":
            self.get_logger().info("Executing turn right")
            # Turn right in an arc
            for _ in range(1):
                twist.linear.x = 0.1
                twist.angular.z = -0.5
                self.pub_cmd_vel.publish(twist)
                rclpy.spin_once(self, timeout_sec=0.1)
        elif sign_name == "repair_work":
            if not self.lidar_active:
                self.get_logger().info("Executing repair_work")
                for _ in range(1):
                    twist.linear.x = 0.30
                    twist.angular.z = 0.0
                    self.pub_cmd_vel.publish(twist)
                    # rclpy.spin_once(self, timeout_sec=0.2)
                    time.sleep(3.0)
                    # twist.linear.x = 0.0
                    # twist.angular.z = 4.5
                    # self.pub_cmd_vel.publish(twist)
                    # # rclpy.spin_once(self, timeout_sec=0.2)
                    # time.sleep(2.5)
                    self.lidar_active = True
        elif sign_name == "parking":
            self.lidar_active = False
            time.sleep(1.2)
            self.parking_active = False  # Активируем парковку
            self.set_active_node('lane_follower_yellow 100')
            return
        elif sign_name == "cross_walk":
            self.lidar_active = True
            self.set_active_node('traffic_sign_detector 100')
        elif sign_name == "tonnel":
            time.sleep(1.0)
            self.set_active_node('traffic_sign_detector 100')
            time.sleep(2.0)
            self.lidar_active = True  # Активируем использование лидара
            

        # After completion, release control
        if not self.lidar_active:    
            self.set_active_node('lane_follower 100')
            self.get_logger().info(f"active {self.is_active}")


def main(args=None):
    rclpy.init(args=args)
    node = TrafficSignDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
