import os
import argparse
import torch
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
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
    pipeline = rs.pipeline()
    config = rs.config()
    profile = pipeline.start(config)
    device = profile.get_device()
    adv_mode = rs.rs400_advanced_mode(device)
    
    if not adv_mode.is_enabled():
        adv_mode.toggle_advanced_mode(True)
        print("Advanced mode enabled.")

    with open(json_file, 'r') as f:
        json_text = f.read().strip()
        adv_mode.load_json(json_text)

    depth_sensor = device.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale is: {depth_scale} meters per unit")

    depth_stream = profile.get_stream(rs.stream.depth)
    intrinsics_depth = depth_stream.as_video_stream_profile().get_intrinsics()
    fx, fy = intrinsics_depth.fx, intrinsics_depth.fy
    cx, cy = intrinsics_depth.ppx, intrinsics_depth.ppy

    return pipeline, fx, fy, cx, cy, depth_scale

def process_frame(frames, align):
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        return None, None

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    color_image = color_image.astype(np.float32) / 255.0

    return color_image, depth_image

def transform_to_base(point, T_bc):
    """
    Transform a point from camera frame to robot base frame.
    
    Args:
    point (np.array): 3D point in camera frame [x, y, z]
    T_bc (np.array): 4x4 transformation matrix from robot base to camera
    
    Returns:
    np.array: 3D point in robot base frame
    """
    point_homogeneous = np.append(point, 1)
    point_base = np.dot(np.linalg.inv(T_bc), point_homogeneous)
    return point_base[:3]

def demo(cfgs):
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    pipeline, fx, fy, cx, cy, scale = setup_realsense(cfgs.realsense_json)

    xmin, xmax = -0.19, 0.12
    ymin, ymax = 0.02, 0.15
    zmin, zmax = 0.0, 1.0
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    align = rs.align(rs.stream.color)

    # Define the transformation matrix from robot base to camera
    # Replace this with your actual transformation matrix
    #T_bc = np.array([
    #    [0.0, -1.0, 0.0, 0.5],
    #    [1.0, 0.0, 0.0, 0.1],
    #    [0.0, 0.0, 1.0, 0.4],
    #    [0.0, 0.0, 0.0, 1.0]
    #])

    T_bc_R = np.array([[ -0.2634318, -0.9643521,  0.0250766],
  [-0.9643186,  0.2639542,  0.0204412],
  [-0.0263316, -0.0187970, -0.9994766] ])
    
    T_bc_T = np.array([-0.50653362059207, -0.00892369975809501, 0.55229938266492])


    # Construct the T_bc from Rotation and Translation
    T_bc = np.eye(4)  # Start with a 4x4 identity matrix
    T_bc[:3, :3] = T_bc_R  # Set the upper-left 3x3 submatrix to the rotation matrix
    T_bc[:3, 3] = T_bc_T 
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            colors, depths = process_frame(frames, align)

            if colors is None or depths is None:
                print("No valid frames received.")
                continue

            xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
            xmap, ymap = np.meshgrid(xmap, ymap)
            points_z = depths * scale
            points_x = (xmap - cx) / fx * points_z
            points_y = (ymap - cy) / fy * points_z

            mask = (points_z > 0) & (points_z < 1)
            points = np.stack([points_x, points_y, points_z], axis=-1)
            points = points[mask].astype(np.float32)
            colors = colors[mask].astype(np.float32)

            vis.clear_geometries()

            o3d_cloud = o3d.geometry.PointCloud()
            o3d_cloud.points = o3d.utility.Vector3dVector(points)
            o3d_cloud.colors = o3d.utility.Vector3dVector(colors)

            vis_trans_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            o3d_cloud.transform(vis_trans_mat)

            vis.add_geometry(o3d_cloud)

            try:
                gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=True, collision_detection=True)

                if gg is not None and len(gg) > 0:
                    gg = gg.nms().sort_by_score()
                    gg_pick = gg[0:3]
                    print("======="*10)
                    print("Top 3 Grasps:")
                    for i, grasp in enumerate(gg_pick):
                        print(f"Grasp {i+1} score: {grasp.score}")
                        
                        # Transform grasp points to robot base frame
                        grasp_point_camera = grasp.translation
                        grasp_point_base = transform_to_base(grasp_point_camera, T_bc)
                        
                        print(f"Grasp {i+1} point in camera frame: {grasp_point_camera}")
                        print(f"Grasp {i+1} point in robot base frame: {grasp_point_base}")
                    print("======="*10)

                    grippers = gg_pick.to_open3d_geometry_list()
                    for i, gripper in enumerate(grippers):
                        gripper.transform(vis_trans_mat)
                        color = [1, 0, 0] if i == 0 else [0, 1, 0] if i == 1 else [0, 0, 1]
                        gripper.paint_uniform_color(color)
                        vis.add_geometry(gripper)

                else:
                    print("No grasp detected after collision detection.")

            except Exception as e:
                print(f"Error during grasp detection: {e}")

            vis.poll_events()
            vis.update_renderer()

    except KeyboardInterrupt:
        print("Live stream stopped by user.")

    finally:
        pipeline.stop()
        vis.destroy_window()

if __name__ == '__main__':
    cfgs = parse_args()
    demo(cfgs)