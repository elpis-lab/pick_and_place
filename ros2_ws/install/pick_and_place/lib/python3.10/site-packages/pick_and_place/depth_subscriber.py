import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32MultiArray
import sys
import cv2
import numpy as np

# Add the directory containing the module to sys.path
module_path = '/home/sultan/Edge-Grasp-Network/'
if module_path not in sys.path:
    sys.path.append(module_path)
    from edge_grasp import gen_grasp
    #from shawarma import gen_grasp

class DepthSubscriber(Node):

    def __init__(self):
        super().__init__('point_cloud_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.listener_callback,
            10)
            
        self.publisher = self.create_publisher(Float32MultiArray, 'grasp', 10)
        self.bridge = CvBridge()
        self.frame_count = 0
        self.depth_maps = []
        self.max_frame_count = 20
        self.max_count_reached = False

    def listener_callback(self, msg):
        self.frame_count += 1
        if self.frame_count < self.max_frame_count + 1:
            depth_map = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            depth_map = depth_map/1000.0
            depth_map = depth_map.astype(np.float32)
            self.depth_maps.append(depth_map)
        else:
            self.get_logger().info(f'{self.frame_count - 1} frames collected')
            grasp = gen_grasp(np.array(self.depth_maps))
            self.get_logger().info(f'Published {grasp}')
            self.max_count_reached = True

def main(args=None):

    rclpy.init(args=args)
    depth_subscriber = DepthSubscriber()
    
    while depth_subscriber.max_count_reached == False:
        rclpy.spin_once(depth_subscriber)

    depth_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
