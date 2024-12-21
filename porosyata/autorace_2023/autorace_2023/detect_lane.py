import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import time

class LaneFollower(Node):
    def __init__(self):
        super().__init__('lane_follower')
        self.sub_image = self.create_subscription(Image, '/color/image', self.image_callback, 1)
        self.pub_image = self.create_publisher(Image, '/lane_follower/image', 1)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 1)

        self.cvBridge = CvBridge()
        self.green_light_detected = False  # Флаг для отслеживания обнаружения зеленого света
        self.is_active = False  # Флаг активности команды
        self.follow_yellow_only = False  # Флаг для движения только по желтой линии

        # Публикация текущего активного узла
        self.control_state_pub = self.create_publisher(String, '/control/active_node', 1)
        self.control_state_sub = self.create_subscription(String, '/control/active_node', self.control_state_callback, 1)
        
        self.turn_speed = 100

        self.max_linear_speed = 0.25

        # msg = String()
        # msg.data = 'lane_follower 100'  # Возвращаем управление LaneFollower
        # self.control_state_pub.publish(msg)

        
    def set_active_node(self, node_name):
        msg = String()
        msg.data = node_name
        self.control_state_pub.publish(msg)

    def control_state_callback(self, msg):
        msg, turn_speed, speed = msg.data.split()
        if msg == 'lane_follower_yellow':
            self.set_follow_yellow_only(True)
        else:
            self.set_follow_yellow_only(False)
            self.is_active = (msg == 'lane_follower')  # Проверяем, активна ли текущая нода
            self.turn_speed = int(turn_speed)
            self.max_linear_speed = float(speed)

    def image_callback(self, msg_img):
        cv_image_original = self.cvBridge.imgmsg_to_cv2(msg_img, "bgr8")
        # Проверка на зеленый свет
        if not self.green_light_detected:
            self.green_light_detected = self.detect_green_light(cv_image_original)
            
        if self.green_light_detected:
            # Обработка изображения для выделения жёлтой и белой линий
            cv_image_processed = self.process_image(cv_image_original)

            # Определение положения автомобиля относительно центра дороги
            center_offset = self.find_center_offset(cv_image_processed)

            # Отправка команд управления для движения по центру
            self.send_control_command(center_offset)
        else:
            # Если зеленый свет не обнаружен, останавливаем движение
            self.send_control_command(0)

        self.pub_image.publish(self.cvBridge.cv2_to_imgmsg(cv_image_original, "bgr8"))

    def detect_green_light(self, cv_image):
        # Преобразование в HSV
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Выделение зеленого света
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Проверка, есть ли зеленый свет на изображении
        green_pixels = cv2.countNonZero(mask_green)
        if green_pixels > 100:  # Порог для обнаружения зеленого света
            msg = String()
            msg.data = 'lane_follower 100 0.4'  # Возвращаем управление LaneFollower
            self.control_state_pub.publish(msg)
            return True
        return False

    def process_image(self, cv_image):
        # Преобразование в HSV
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Выделение жёлтой линии
        lower_yellow = np.array([20, 100, 230])
        upper_yellow = np.array([30, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Выделение белой линии
        lower_white = np.array([0, 0, 230])
        upper_white = np.array([180, 30, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        yellow_pixels = cv2.countNonZero(mask_yellow)
        # self.get_logger().info(f"pixels: {yellow_pixels}")
        # Объединение масок
        if self.follow_yellow_only:
            # Если движение только по желтой линии, используем только желтую маску
            mask = mask_yellow
            yellow_pixels = cv2.countNonZero(mask_yellow)
            self.get_logger().info(f"pixels: {yellow_pixels}")
            if yellow_pixels < 2000:  # Порог для обнаружения зеленого света
                self.follow_yellow_only = False
                twist = Twist()
                twist.angular.z = -0.5
                twist.linear.x = 0.2
                self.pub_cmd_vel.publish(twist)
                time.sleep(1.5)
                self.set_active_node('parking 50 0.25')
        else:
            # Если движение возможно по обеим линиям, объединяем маски
            mask = cv2.bitwise_or(mask_yellow, mask_white)

        # Применение маски к изображению
        cv_image_processed = cv2.bitwise_and(cv_image, cv_image, mask=mask)

        return cv_image_processed

    def find_center_offset(self, cv_image):
        # Находим центр изображения
        height, width, _ = cv_image.shape
        center_x = width // 2
        lower_yellow = np.array([20, 100, 230])
        upper_yellow = np.array([30, 255, 255])

        lower_white = np.array([0, 0, 230])
        upper_white = np.array([180, 30, 255])

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        if self.follow_yellow_only:
            # Для флага follow_yellow_only: анализируем только левую сторону изображения
            left_half = cv_image[:, :center_x]
            yellow_center = self.find_line_center(left_half, lower_yellow, upper_yellow)
            self.get_logger().info(f"center: {yellow_center}")
            if not yellow_center:
                yellow_center = 0
            center_offset = yellow_center - center_x
            center_offset+= 260
        else:
            # Для обычного режима анализируем обе стороны
            left_half = cv_image[:, :center_x]
            right_half = cv_image[:, center_x:]

            yellow_center = self.find_line_center(left_half, lower_yellow, upper_yellow)
            white_center = self.find_line_center(right_half, lower_white, upper_white)

            # Смещаем координаты белой линии в общую систему координат
            if white_center is not None:
                white_center += center_x
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            yellow_pixels = cv2.countNonZero(mask_yellow)
            if yellow_pixels < 1500:
                yellow_center = None
            mask_white = cv2.inRange(hsv, lower_white, upper_white)
            white_pixels = cv2.countNonZero(mask_white)
            if white_pixels < 4500:
                white_center = None
            # Вычисляем смещение от центра дороги
            # self.get_logger().info(f"yellow_center: {yellow_center} white_center {white_center} {white_pixels}")
            if yellow_center is not None and white_center is not None:
                road_center = (yellow_center + white_center) // 2
                center_offset = road_center - center_x
            elif yellow_center is not None:
                # center_offset = yellow_center - center_x  # Только жёлтая линия
                center_offset = 100
            elif white_center is not None:
                # center_offset = white_center - center_x  # Только белая линия
                center_offset = -100
            else:
                center_offset = 0

        return center_offset


    def find_line_center(self, cv_image, lower_color, upper_color):
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color, upper_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                return cx
        return None

    # def send_control_command(self, center_offset):
    #     if self.follow_yellow_only:
    #         center_offset+= 255
    #     # self.get_logger().info(f"center: {center_offset}")
    #     if not self.is_active:  # Проверяем, не занято ли управление
    #         return  # Если другая нода управляет движением, пропускаем
    #     twist = Twist()
    #     if self.green_light_detected:
    #         twist.linear.x = 0.15  # Устанавливаем линейную скорость
    #         twist.angular.z = -float(center_offset) / self.turn_speed  # Устанавливаем угловую скорость в зависимости от смещения
    #     else:
    #         twist.linear.x = 0.0  # Останавливаем движение, если зеленый свет не обнаружен
    #         twist.angular.z = 0.0
    #     self.pub_cmd_vel.publish(twist)

    def send_control_command(self, center_offset):
        # self.get_logger().info(f"activ: {self.is_active}")
        if not self.is_active:  # Проверяем, не занято ли управление
            return  # Если другая нода управляет движением, пропускаем
        # self.get_logger().info(f"center: {center_offset}")
        twist = Twist()

        # Рассчитываем угловую скорость
        angular_speed = -float(center_offset-20) / self.turn_speed

        # Параметры для управления скоростью
        min_linear_speed = 0.01  # Минимальная линейная скорость на крутых поворотах
        max_angular_speed = 2.0  # Максимальная угловая скорость для интерполяции

        # Коэффициент влияния угловой скорости на замедление
        slowdown_factor = 1.0  # Чем больше значение, тем сильнее замедление при поворотах

        # Масштабирование линейной скорости: нелинейная интерполяция
        normalized_angular = min(abs(angular_speed) / max_angular_speed, 1.0)
        twist.linear.x = self.max_linear_speed - (normalized_angular ** slowdown_factor) * (self.max_linear_speed - min_linear_speed)
        # if abs(center_offset) > 60:  # Если смещение превышает определенный порог
        #     twist.linear.x = 0.0  # Останавливаем линейное движение
        # Устанавливаем угловую скорость
        twist.angular.z = angular_speed

        # Публикуем команду
        self.pub_cmd_vel.publish(twist)


    def set_follow_yellow_only(self, value):
        # Переключаем флаг движения только по желтой линии
        self.is_active = value
        self.follow_yellow_only = value
        self.get_logger().info(f"Follow yellow only: {self.follow_yellow_only}")


def main(args=None):    
    rclpy.init(args=args)   
    node = LaneFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
