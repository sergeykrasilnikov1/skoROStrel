import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class CmdVelLogger(Node):
    def __init__(self):
        super().__init__('cmd_vel_logger')
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10)
        self.subscription  # prevent unused variable warning

    def cmd_vel_callback(self, msg):
        self.get_logger().info(f"Linear: x={msg.linear.x}, y={msg.linear.y}, z={msg.linear.z}")
        self.get_logger().info(f"Angular: x={msg.angular.x}, y={msg.angular.y}, z={msg.angular.z}")

def main(args=None):
    rclpy.init(args=args)
    node = CmdVelLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()