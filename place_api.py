import pyrealsense2 as rs
from list_realsense import list_realsense_cameras
from transforms3d.quaternions import mat2quat, quat2mat
import numpy as np
import traceback
import cv2
from fastapi import FastAPI
import requests
from PIL import Image
import io
from typing import Tuple, Dict
import base64
import json
from pydantic import BaseModel

def encode_image_to_base64(image_array: np.ndarray) -> str:
    """Convert numpy image array to base64 string"""
    success, encoded_image = cv2.imencode('.png', image_array)
    if not success:
        return None
    return base64.b64encode(encoded_image.tobytes()).decode('utf-8')

class Place_API:
    def __init__(self):
        # Initialize camera-related variables
        self.pipeline = None
        self.align = None
        self.depth_scale = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # Segmentation API endpoint
        self.segment_api_url = "http://localhost:8004/segment"

        # Get id of the realsense D435
        cameras = list_realsense_cameras()
        print(f"Found {len(cameras)} RealSense camera(s):")
        for i, camera in enumerate(cameras, 1):
            if "D455F" in camera['name']:
                self.realsense_id = i
                self.realsense_serial_no = camera['serial_number']  
                break
        
        # Setup RealSense camera
        self.setup_realsense("realsense_config.json")
        
        # Transform matrix from base to camera
        self.temp_xyz = [-0.605256,-0.336046, 1.04207]
        self.temp_xyzw = [0.711241, -0.702776, -0.0126768, 0.00905347]
        if self.temp_xyz and self.temp_xyzw:
            self.T_base_camera = np.eye(4)
            self.T_base_camera[:3, 3] = self.temp_xyz
            self.T_base_camera[:3, :3] = quat2mat(self.temp_xyzw)
        else:
            self.T_base_camera = np.eye(4)

    def setup_realsense(self, json_file: str):
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            
            print(f"Connecting to camera with serial: {self.realsense_serial_no}")
            config.enable_device(self.realsense_serial_no)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            profile = pipeline.start(config)
            device = profile.get_device()
            
            # Get depth scale
            depth_sensor = device.first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            # Get camera intrinsics
            depth_stream = profile.get_stream(rs.stream.depth)
            intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
            self.fx = intrinsics.fx
            self.fy = intrinsics.fy
            self.cx = intrinsics.ppx
            self.cy = intrinsics.ppy
            
            self.pipeline = pipeline
            self.align = rs.align(rs.stream.color)
            
            print("Camera setup complete")
            print(f"Intrinsics: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
            print(f"Depth scale: {self.depth_scale}")
            
        except Exception as e:
            print(f"Error setting up camera: {e}")
            traceback.print_exc()
            raise

    def pixel_to_3d(self, pixel_x: int, pixel_y: int, depth: float) -> np.ndarray:
        x = (pixel_x - self.cx) * depth / self.fx
        y = (pixel_y - self.cy) * depth / self.fy
        z = depth
        return np.array([x, y, z])
    
    def project_3d_to_2d(self, point_3d):
        """Project 3D point to 2D image coordinates"""
        x, y, z = point_3d
        u = (x * self.fx / z) + self.cx
        v = (y * self.fy / z) + self.cy
        return np.array([u, v])

    def camera_to_base_transform(self, point_camera: np.ndarray) -> np.ndarray:
        point_homogeneous = np.append(point_camera, 1)
        point_base = self.T_base_camera @ point_homogeneous
        return point_base[:3]

    def get_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        depth_image = np.asanyarray(depth_frame.get_data()) * self.depth_scale
        color_image = np.asanyarray(color_frame.get_data())
        
        return color_image, depth_image

    def draw_segmentation(self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Draw segmentation mask on image with transparency"""
        mask_colored = np.zeros_like(image)
        mask_colored[mask > 0] = [0, 255, 0]  # Green mask
        return cv2.addWeighted(image, 1, mask_colored, alpha, 0)

    def estimate_rotation(self, mask: np.ndarray, depth_image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Estimate rotation from mask and depth data using PCA"""
        y_coords, x_coords = np.where(mask)
        if len(x_coords) < 10:
            return np.eye(3), 0.0
        
        points_3d = []
        for x, y in zip(x_coords, y_coords):
            depth = depth_image[y, x]
            if depth > 0:  # Filter out invalid depth
                point_3d = self.pixel_to_3d(x, y, depth)
                points_3d.append(point_3d)
        
        if len(points_3d) < 10:
            return np.eye(3), 0.0
        
        points_3d = np.array(points_3d)
        
        # Perform PCA
        mean = np.mean(points_3d, axis=0)
        points_centered = points_3d - mean
        cov = np.cov(points_centered.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        # Sort by eigenvalues in descending order
        idx = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Create rotation matrix ensuring right-hand coordinate system
        R = np.column_stack([eigenvecs[:, 0],
                           np.cross(eigenvecs[:, 0], eigenvecs[:, 2]),
                           eigenvecs[:, 2]])
        
        # Ensure proper rotation matrix
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1
            
        # Calculate confidence based on eigenvalue ratios
        confidence = min(1.0, eigenvals[0] / (eigenvals[1] + 1e-6))
        
        return R, confidence

    def calculate_transforms(self, point_camera: np.ndarray, mask: np.ndarray, depth_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Calculate all transforms: camera to object and base to object
        
        Returns:
            T_camera_obj: Transform from camera to object
            T_base_obj: Transform from base to object
            pos_base: Position in base frame
            quat_base: Orientation in base frame (as quaternion)
            confidence: Confidence in orientation estimation
        """
        # Get rotation in camera frame
        R_camera_obj, confidence = self.estimate_rotation(mask, depth_image)
        
        # Create camera to object transform
        T_camera_obj = np.eye(4)
        T_camera_obj[:3, :3] = R_camera_obj
        T_camera_obj[:3, 3] = point_camera

        # Invert the transform of current base to camera to get camera to base
        T_base_camera = np.linalg.inv(self.T_base_camera)
        #T_base_camera = self.T_base_camera
        
        # Calculate base to object transform
        T_base_obj = T_base_camera @ T_camera_obj
        
        # Extract position and rotation for base frame
        pos_base = T_base_obj[:3, 3]
        rot_base = T_base_obj[:3, :3]
        quat_base = mat2quat(rot_base)
        
        return T_camera_obj, T_base_obj, pos_base, quat_base, confidence
    
    def transform_point_camera_to_base(self, point_camera: np.ndarray) -> np.ndarray:
        """
        Transform a point from camera frame to base frame using matrix multiplication
        
        Args:
            point_camera: Point in camera frame [x, y, z]
        
        Returns:
            pos_base: Position in base frame [x, y, z]
        """
        # Get rotation and translation from camera to base
        #T_base_camera = np.linalg.inv(self.T_base_camera)
        T_base_camera = self.T_base_camera
        # temp_rotation
        temp_rot =[[0,1], [1,0]]
        T_base_cam_xy = T_base_camera[:2, 3]

        point_cam_xy = point_camera[:2]
        #print("Point Camera XY: ", point_cam_xy)
        #print("T_base_cam_xy: ", T_base_cam_xy)
        #print("Temp Rotation: ", temp_rot)

        pos_base = temp_rot @ point_cam_xy - T_base_cam_xy
        pos_base = np.append(pos_base, point_camera[2])
        #print("Pos Base: ", pos_base)
        
        return pos_base

    # Modify your process_frame method to include the transforms:
    def process_frame(self, prompt: str) -> Dict:
        # Get frames
        color_image, depth_image = self.get_frame()
        
        # Prepare images for API
        _, rgb_image = cv2.imencode('.jpg', color_image)
        
        # API request
        files = {
            'rgb_image': ('image.jpg', rgb_image.tobytes(), 'image/jpeg')
        }
        data = {'prompt': prompt}
        
        try:
            response = requests.post(self.segment_api_url, files=files, data=data)
            #print("API response:", response.status_code)
            #print("API response:", response.json()) 
            results = response.json()['results']
            
            processed_results = []
            visualization = color_image.copy()
            
            for result in results:
                if result['center'] is None:
                    continue
                    
                cX, cY = result['center']
                depth = depth_image[cY, cX]
                mask = np.array(result['mask'])
                
                # Get 3D position in camera frame
                pos_camera = self.pixel_to_3d(cX, cY, depth)
                
                # Calculate all transforms
                T_camera_obj, T_base_obj, pos_base, quat_base, rot_confidence = self.calculate_transforms(
                    pos_camera, mask, depth_image
                )

                #print("Pose Base before transform: ", pos_base)
                pos_base = self.transform_point_camera_to_base(pos_camera)
                #print("Pose Base after transform: ", pos_base)

                
                processed_result = {
                    "phrase": result['phrase'],
                    "confidence": float(result['confidence']),
                    "rotation_confidence": float(rot_confidence),
                    "center_pixel": [int(cX), int(cY)],
                    "depth": float(depth),
                    "base_to_camera_transform": self.T_base_camera.tolist(),
                    "position_camera": pos_camera.tolist(),
                    "position_base": pos_base.tolist(),
                    "orientation_base": quat_base.tolist(),  # [w, x, y, z] format
                    "transform_camera_obj": T_camera_obj.tolist(),
                    "transform_base_obj": T_base_obj.tolist(),
                    "bounding_box": result['bounding_box']
                }
                processed_results.append(processed_result)
                
                # Add segmentation mask
                visualization = self.draw_segmentation(visualization, mask)
                
                # Draw bounding box
                box = result['bounding_box']
                cv2.rectangle(visualization, 
                            (int(box[0]), int(box[1])), 
                            (int(box[2]), int(box[3])), 
                            (0, 255, 0), 2)
                
                # Draw center point
                cv2.circle(visualization, (cX, cY), 5, (255, 0, 0), -1)
                
                # Draw coordinate axes to show orientation
                axis_length = 30
                origin = np.array([cX, cY])
                rotation_2d = T_camera_obj[:2, :2]  # Use only 2D part of rotation
                
                # Draw axes
                for i, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):  # RGB for XYZ
                    direction = rotation_2d @ np.array([1 if j == i else 0 for j in range(2)])
                    endpoint = origin + (direction * axis_length).astype(int)
                    cv2.line(visualization, tuple(origin.astype(int)), tuple(endpoint), color, 2)
                
                # Add text for pose
                text_pos = f"Pos: {pos_base[0]:.2f}, {pos_base[1]:.2f}, {pos_base[2]:.2f}"
                text_rot = f"Rot: {quat_base[0]:.2f}, {quat_base[1]:.2f}, {quat_base[2]:.2f}, {quat_base[3]:.2f}"
                cv2.putText(visualization, text_pos, 
                           (int(box[0]), int(box[1])-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(visualization, text_rot, 
                           (int(box[0]), int(box[1])-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return {
                "results": processed_results,
                "visualization_base64": encode_image_to_base64(visualization)
            }
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            traceback.print_exc()
            return {"error": str(e)}
# FastAPI setup
app = FastAPI()
place_api = Place_API()


class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Place API"}

@app.post("/process")
async def process_endpoint(request: PromptRequest):
    print("Given String: ", request.prompt)
    try:
        return place_api.process_frame(request.prompt)
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)