import pyrealsense2 as rs
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32MultiArray
import sys
import cv2
import json

# Add the directory containing the module to sys.path
module_path = '/home/sultan/Edge-Grasp-Network/'
if module_path not in sys.path:
    sys.path.append(module_path)
    from edge_grasp import gen_grasp
    #from shawarma import gen_grasp

class GraspGenerator(Node):

    def __init__(self):
        super().__init__('grasp_gen')
        self.publisher = self.create_publisher(Float32MultiArray, 'grasp', 10)
        self.max_frame_count = 1
        print('reached1')
        self.publisher_callback()

    def publisher_callback(self):
        print('reached2')
        stacked_depth_maps = self.get_depth_maps(num_depth_imgs=self.max_frame_count)
        grasp = gen_grasp(stacked_depth_maps).astype(np.float32).flatten().tolist()

        if grasp:
            msg = Float32MultiArray()
            msg.data = grasp
            self.publisher.publish(msg)
            self.get_logger().info(f'Published {grasp}')
        else:
            self.get_logger().info(f'No gasp found.')

    
    def get_depth_maps(self, num_depth_imgs=1, json_file_path='/home/sultan/Edge-Grasp-Network/realsense_config.json'):
    
        # Create a pipeline
        pipeline = rs.pipeline()

        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        config = rs.config()

        # Load the JSON settings
        jsonObj = json.load(open(json_file_path))
        json_string= str(jsonObj).replace("'", '\"')

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg = pipeline.start(config)
        dev = cfg.get_device()
        advnc_mode = rs.rs400_advanced_mode(dev)
        advnc_mode.load_json(json_string)

        # Get depth scale
        depth_sensor = cfg.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        align = rs.align(align_to)

        stacked_depth_maps = []
        i = 0
        # Streaming loop
        try:
            while i < num_depth_imgs:

                # Get frameset of color and depth
                frames = pipeline.wait_for_frames()
                # frames.get_depth_frame() is a 640x360 depth image

                # Align the depth frame to color frame
                aligned_frames = align.process(frames)

                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image

                # Validate that both frames are valid
                if not aligned_depth_frame:
                    continue
                else:
                    i += 1 # Count frame only if it is valid

                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                depth_map = depth_image * depth_scale
                stacked_depth_maps.append(depth_map)

        finally:
            pipeline.stop()
            return np.array(stacked_depth_maps, dtype=np.float32)

def main(args=None):

    rclpy.init(args=args)
    grasp_gen = GraspGenerator()

    rclpy.spin_once(grasp_gen)
    grasp_gen.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
