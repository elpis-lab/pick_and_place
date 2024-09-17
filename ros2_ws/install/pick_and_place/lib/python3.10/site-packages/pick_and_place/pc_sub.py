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

class PointCloudSubscriber(Node):

    def __init__(self):
        super().__init__('point_cloud_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.listener_callback,
            10)
            
        self.publisher = self.create_publisher(Float32MultiArray, 'grasp', 10)
        self.depth_map_count = 0
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        
        depth_map = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        grasp = gen_grasp(depth_map)
        #new_msg = Float32MultiArray()
        #new_msg.data = grasp
        #self.publisher.publish(new_msg)
        self.get_logger().info(f'Published')
        self.depth_map_count += 1

def main(args=None):

    rclpy.init(args=args)
    point_cloud_subscriber = PointCloudSubscriber()
    
    while point_cloud_subscriber.depth_map_count < 1:
      rclpy.spin_once(point_cloud_subscriber)
      
    point_cloud_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
