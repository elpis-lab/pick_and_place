from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import torch
import open3d as o3d
from PIL import Image
import io
import yaml
from pydantic import BaseModel

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

app = FastAPI()

# Initialize AnyGrasp and config
anygrasp = None
config = None

class Config(BaseModel):
    model: dict
    camera: dict
    workspace: dict
    grasp: dict

def load_config(file_path: str) -> Config:
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return Config(**config_dict)

@app.on_event("startup")
async def startup_event():
    global anygrasp, config
    config = load_config('config.yaml')
    
    class ModelConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    model_config = ModelConfig(**config.model)
    anygrasp = AnyGrasp(model_config)
    anygrasp.load_net()

# Generic endpoint to check if the API is running
@app.get("/")
async def read_root():
    return {"message": "Welcome to the AnyGrasp API! - Grasp Detection"}

@app.post("/get_grasp")
async def get_grasp(
    color_image: UploadFile = File(...),
    depth_image: UploadFile = File(...)
):
    # Read and process the uploaded images
    color_data = await color_image.read()
    depth_data = await depth_image.read()
    
    colors = np.array(Image.open(io.BytesIO(color_data)), dtype=np.float32) / 255.0
    depths = np.array(Image.open(io.BytesIO(depth_data)))

    # Get camera intrinsics from config
    fx, fy = config.camera['fx'], config.camera['fy']
    cx, cy = config.camera['cx'], config.camera['cy']
    scale = config.camera['scale']

    # Set workspace to filter output grasps
    lims = [
        config.workspace['xmin'], config.workspace['xmax'],
        config.workspace['ymin'], config.workspace['ymax'],
        config.workspace['zmin'], config.workspace['zmax']
    ]

    # Get point cloud
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # Set your workspace to crop point cloud
    mask = (points_z > 0) & (points_z < 1)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)

    gg, cloud = anygrasp.get_grasp(
        points, colors, lims=lims, 
        apply_object_mask=config.grasp['apply_object_mask'],
        dense_grasp=config.grasp['dense_grasp'],
        collision_detection=config.grasp['collision_detection']
    )

    if gg is None:
        gg = []
    if len(gg) == 0 or gg is None:
        return JSONResponse(content={"message": "No Grasp detected after collision detection!"}, status_code=404)

    gg = gg.nms().sort_by_score()
    gg_pick = gg[0:config.grasp['num_grasps']]

    # Convert grasp data to a serializable format
    grasp_data = []
    for grasp in gg_pick:
        grasp_data.append({
            "score": float(grasp.score),
            "translation": grasp.translation.tolist(),
            "rotation_matrix": grasp.rotation_matrix.tolist(),
            "width": float(grasp.width),
            "depth": float(grasp.depth)
        })
 
    return {"grasps": grasp_data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)