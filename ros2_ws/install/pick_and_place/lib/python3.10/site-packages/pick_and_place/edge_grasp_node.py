import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32MultiArray


class EdgeGrasp(Node):

    def __init__(self):
        super().__init__('edge_grasp_sub_pub')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'topic',
            self.listener_callback,
            10)
        self.subscription

        self.publisher = self.create_publisher(Pose, 'topic', 10)
    
    def publisher_callback(self, x, y, z, qx, qy, qz, qw):

        msg = Pose()

        msg.data.position.x = x
        msg.data.position.y = y
        msg.data.position.z = z
        msg.data.orientation.qx = qx
        msg.data.orientation.qy = qy
        msg.data.orientation.qz = qz
        msg.data.orientation.qw = qw

        self.get_logger().info('The Grasp is : "%s"' % msg.data)


    def listener_callback(self, msg):

        depth_map = msg.data

        

        self.get_logger().info('Depth Map received.')


def main(args=None):
    rclpy.init(args=args)

    edge_grasp_sub_pub = EdgeGrasp()

    rclpy.spin(edge_grasp_sub_pub)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    edge_grasp_sub_pub.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()