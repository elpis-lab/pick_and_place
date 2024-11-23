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
from typing import Tuple, Dict, List, Optional
import base64
import json
from pydantic import BaseModel
from dataclasses import dataclass
import time

def encode_image_to_base64(image_array: np.ndarray) -> str:
    """Convert numpy image array to base64 string"""
    success, encoded_image = cv2.imencode('.png', image_array)
    if not success:
        return None
    return base64.b64encode(encoded_image.tobytes()).decode('utf-8')

@dataclass
class CameraConfig:
    serial_number: str
    transform_base_camera: np.ndarray  # 4x4 transformation matrix

class DualCameraPlaceAPI:
    def __init__(self):
        # Initialize camera configs
        self.cameras: Dict[str, CameraConfig] = {}
        self.pipelines: Dict[str, rs.pipeline] = {}
        self.aligners: Dict[str, rs.align] = {}
        self.intrinsics: Dict[str, Dict[str, float]] = {}
        
        # Segmentation API endpoint
        self.segment_api_url = "http://localhost:8004/segment"
        
        # Find D455F cameras
        self._setup_cameras()
        
    def _setup_cameras(self):
        cameras = list_realsense_cameras()
        print(f"Found {len(cameras)} RealSense camera(s):")
        
        d455f_cameras = [cam for cam in cameras if "D455F" in cam['name']]
        if len(d455f_cameras) < 2:
            raise RuntimeError(f"Need 2 D455F cameras, found {len(d455f_cameras)}")
            
        # Example transforms - replace these with your actual extrinsics
        transforms = {
            d455f_cameras[0]['serial_number']: np.array([
                [-0.605256, -0.336046, 1.04207, 0],
                [0.711241, -0.702776, -0.0126768, 0],
                [0.00905347, 0, 0, 1],
                [0, 0, 0, 1]
            ]),
            d455f_cameras[1]['serial_number']: np.array([
                # Replace with second camera transform
                [-0.605256, -0.336046, 1.04207, 0],
                [0.711241, -0.702776, -0.0126768, 0],
                [0.00905347, 0, 0, 1],
                [0, 0, 0, 1]
            ])
        }
        
        for camera in d455f_cameras[:2]:  # Take first two D455F cameras
            serial = camera['serial_number']
            self.cameras[serial] = CameraConfig(
                serial_number=serial,
                transform_base_camera=transforms[serial]
            )
            self._setup_single_camera(serial)


    # 336522303434

    def _setup_single_camera(self, serial_number: str):
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            
            print(f"Connecting to camera with serial: {serial_number}")
            config.enable_device(serial_number)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            profile = pipeline.start(config)
            device = profile.get_device()
            
            # Get depth scale and intrinsics
            depth_sensor = device.first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            
            depth_stream = profile.get_stream(rs.stream.depth)
            intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
            
            self.pipelines[serial_number] = pipeline
            self.aligners[serial_number] = rs.align(rs.stream.color)
            self.intrinsics[serial_number] = {
                'fx': intrinsics.fx,
                'fy': intrinsics.fy,
                'cx': intrinsics.ppx,
                'cy': intrinsics.ppy,
                'depth_scale': depth_scale
            }
            
            print(f"Camera {serial_number} setup complete")
            print(f"Intrinsics: {self.intrinsics[serial_number]}")
            
        except Exception as e:
            print(f"Error setting up camera {serial_number}: {e}")
            traceback.print_exc()
            raise

    def pixel_to_3d(self, pixel_x: int, pixel_y: int, depth: float, serial_number: str) -> np.ndarray:
        intr = self.intrinsics[serial_number]
        x = (pixel_x - intr['cx']) * depth / intr['fx']
        y = (pixel_y - intr['cy']) * depth / intr['fy']
        z = depth
        return np.array([x, y, z])

    def get_frame(self, serial_number: str) -> Tuple[np.ndarray, np.ndarray]:
        frames = self.pipelines[serial_number].wait_for_frames()
        aligned_frames = self.aligners[serial_number].process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        depth_image = np.asanyarray(depth_frame.get_data()) * self.intrinsics[serial_number]['depth_scale']
        color_image = np.asanyarray(color_frame.get_data())
        
        return color_image, depth_image

    def transform_point_camera_to_base(self, point_camera: np.ndarray, serial_number: str) -> np.ndarray:
        T_base_camera = self.cameras[serial_number].transform_base_camera
        temp_rot = [[0, 1], [1, 0]]
        T_base_cam_xy = T_base_camera[:2, 3]
        point_cam_xy = point_camera[:2]
        pos_base = temp_rot @ point_cam_xy - T_base_cam_xy
        return np.append(pos_base, point_camera[2])

    def process_dual_cameras(self, prompt: str) -> Dict:
        all_results = {}
        visualizations = {}
        
        # Process each camera
        for serial_number in self.cameras.keys():
            color_image, depth_image = self.get_frame(serial_number)
            _, rgb_image = cv2.imencode('.jpg', color_image)
            
            # Make API request
            try:
                response = requests.post(
                    self.segment_api_url,
                    files={'rgb_image': ('image.jpg', rgb_image.tobytes(), 'image/jpeg')},
                    data={'prompt': prompt}
                )
                results = response.json()['results']
                
                processed_results = []
                visualization = color_image.copy()
                
                for result in results:
                    if result['center'] is None:
                        continue
                        
                    cX, cY = result['center']
                    depth = depth_image[cY, cX]
                    
                    # Get 3D position
                    pos_camera = self.pixel_to_3d(cX, cY, depth, serial_number)
                    pos_base = self.transform_point_camera_to_base(pos_camera, serial_number)
                    
                    processed_result = {
                        "phrase": result['phrase'],
                        "confidence": float(result['confidence']),
                        "position_base": pos_base.tolist(),
                        "camera_serial": serial_number
                    }
                    processed_results.append(processed_result)
                    
                    # Visualization
                    mask = np.array(result['mask'])
                    cv2.circle(visualization, (cX, cY), 5, (255, 0, 0), -1)
                    text_pos = f"Pos: {pos_base[0]:.2f}, {pos_base[1]:.2f}, {pos_base[2]:.2f}"
                    cv2.putText(visualization, text_pos, 
                              (cX, cY-10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 255, 0), 2)
                
                all_results[serial_number] = processed_results
                visualizations[serial_number] = visualization
                
            except Exception as e:
                print(f"Error processing camera {serial_number}: {e}")
                traceback.print_exc()
                return {"error": str(e)}
        
        # Compare results between cameras
        matching_objects = self._find_matching_objects(all_results)
        
        return {
            "individual_results": all_results,
            "matching_objects": matching_objects,
            "visualizations": {
                serial: encode_image_to_base64(vis) 
                for serial, vis in visualizations.items()
            }
        }

    def _find_matching_objects(self, all_results: Dict[str, List[Dict]]) -> List[Dict]:
        matching_objects = []
        position_threshold = 0.1  # 10cm threshold for matching positions
        
        # Get camera serials
        camera_serials = list(all_results.keys())
        if len(camera_serials) < 2:
            return []
            
        # Compare objects between cameras
        for obj1 in all_results[camera_serials[0]]:
            for obj2 in all_results[camera_serials[1]]:
                if obj1['phrase'] == obj2['phrase']:
                    pos1 = np.array(obj1['position_base'])
                    pos2 = np.array(obj2['position_base'])
                    
                    # Check if positions are close enough
                    if np.linalg.norm(pos1 - pos2) < position_threshold:
                        # Average the positions
                        avg_position = (pos1 + pos2) / 2
                        matching_objects.append({
                            "phrase": obj1['phrase'],
                            "position": avg_position.tolist(),
                            "confidence": (obj1['confidence'] + obj2['confidence']) / 2,
                            "detected_by": [camera_serials[0], camera_serials[1]]
                        })
                        break
            else:
                # Object only detected by first camera
                matching_objects.append({
                    "phrase": obj1['phrase'],
                    "position": obj1['position_base'],
                    "confidence": obj1['confidence'],
                    "detected_by": [camera_serials[0]]
                })
        
        # Add objects only detected by second camera
        detected_phrases = {obj['phrase'] for obj in matching_objects}
        for obj2 in all_results[camera_serials[1]]:
            if obj2['phrase'] not in detected_phrases:
                matching_objects.append({
                    "phrase": obj2['phrase'],
                    "position": obj2['position_base'],
                    "confidence": obj2['confidence'],
                    "detected_by": [camera_serials[1]]
                })
        
        return matching_objects

# FastAPI setup
app = FastAPI()
place_api = DualCameraPlaceAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/process_dual")
async def process_dual_endpoint(request: PromptRequest):
    try:
        return place_api.process_dual_cameras(request.prompt)
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)