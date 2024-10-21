import os
import argparse
import torch
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from PIL import Image
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
    parser.add_argument('--realsense_json', required=True, help='RealSense JSON file for pipeline configuration')
    parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
    parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
    parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    args.max_gripper_width = max(0, min(0.1, args.max_gripper_width))
    return args

def setup_realsense(json_file):
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Start the pipeline with default configuration
    profile = pipeline.start(config)

    # Get the RealSense device
    device = profile.get_device()

    # Enable advanced mode
    adv_mode = rs.rs400_advanced_mode(device)

    if not adv_mode.is_enabled():
        adv_mode.toggle_advanced_mode(True)
        print("Advanced mode enabled.")

    # Load JSON configuration
    with open(json_file, 'r') as f:
        json_text = f.read().strip()
        adv_mode.load_json(json_text)

    # Get depth sensor's depth scale
    depth_sensor = device.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale is: {depth_scale} meters per unit")

    # Get intrinsics for the depth stream
    depth_stream = profile.get_stream(rs.stream.depth)
    intrinsics_depth = depth_stream.as_video_stream_profile().get_intrinsics()
    fx, fy = intrinsics_depth.fx, intrinsics_depth.fy
    cx, cy = intrinsics_depth.ppx, intrinsics_depth.ppy

    return pipeline, fx, fy, cx, cy, depth_scale

def process_frame(frames, align):
    # Align depth frame to color frame
    aligned_frames = align.process(frames)

    # Extract aligned depth and color frames
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        return None, None

    # Convert frames to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Normalize the color image
    color_image = color_image.astype(np.float32) / 255.0

    return color_image, depth_image

def demo(cfgs):
    # Initialize the AnyGrasp model
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # Setup RealSense pipeline with JSON configuration
    pipeline, fx, fy, cx, cy, scale = setup_realsense(cfgs.realsense_json)

    # Set workspace limits
    xmin, xmax = -0.19, 0.12
    ymin, ymax = 0.02, 0.15
    zmin, zmax = 0.0, 1.0
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Align the depth stream to the color stream
    align = rs.align(rs.stream.color)

    try:
        while True:
            # Wait for new set of frames from the RealSense camera
            frames = pipeline.wait_for_frames()
            colors, depths = process_frame(frames, align)

            if colors is None or depths is None:
                print("No valid frames received.")
                continue

            # Generate point cloud from depth
            xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
            xmap, ymap = np.meshgrid(xmap, ymap)
            points_z = depths * scale  # Use the real depth scale
            points_x = (xmap - cx) / fx * points_z
            points_y = (ymap - cy) / fy * points_z

            # Mask valid points
            mask = (points_z > 0) & (points_z < 1)
            points = np.stack([points_x, points_y, points_z], axis=-1)
            points = points[mask].astype(np.float32)
            colors = colors[mask].astype(np.float32)

            # Clear previous geometries
            vis.clear_geometries()

            # Create and transform point cloud
            trans_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            o3d_cloud = o3d.geometry.PointCloud()
            o3d_cloud.points = o3d.utility.Vector3dVector(points)
            o3d_cloud.colors = o3d.utility.Vector3dVector(colors)
            o3d_cloud.transform(trans_mat)

            # Add point cloud to visualizer
            vis.add_geometry(o3d_cloud)

            best_pose = None
            try:
                gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=True, collision_detection=True)

                if gg is not None and len(gg) > 0:
                    gg = gg.nms().sort_by_score()
                    gg_pick = gg[0:3]   # Taking the top 3 only
                    print("======="*10)
                    print("Top Grasps", gg_pick)
                    print('Top grasp score:', gg_pick[0].score)
                    print("======="*10)

                    # Get the best pose (highest scoring grasp)
                    best_pose = gg_pick[0]

                    # Visualize grippers
                    grippers = gg_pick.to_open3d_geometry_list()
                    for i, gripper in enumerate(grippers):
                        gripper.transform(trans_mat)
                        # Assign different colors to top 3 grippers 
                        color = [1,0,0] if i == 0 else [0,1,0] if i == 1 else [0,0,1]
                        gripper.paint_uniform_color(color)
                        vis.add_geometry(gripper) 
                    #for gripper in grippers:
                    #    gripper.transform(trans_mat)
                    #    vis.add_geometry(gripper)

                else:
                    print("No grasp detected after collision detection.")

            except Exception as e:
                print(f"Error during grasp detection: {e}")

            # Update visualization
            vis.poll_events()
            vis.update_renderer()

            # If you want to use the best_pose for further processing, you can do it here
            if best_pose:
                print(f"Best pose: Translation: {best_pose.translation}, Rotation: {best_pose.rotation_matrix}")

    except KeyboardInterrupt:
        print("Live stream stopped by user.")

    finally:
        # Stop the RealSense pipeline and destroy the window
        pipeline.stop()
        vis.destroy_window()

if __name__ == '__main__':
    cfgs = parse_args()
    demo(cfgs)