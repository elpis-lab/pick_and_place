from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
from lang_sam import LangSAM
import cv2

app = FastAPI()

# Initialize LangSAM model
model = LangSAM()

def get_segment_center(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX, cY)
    
    return None

@app.get("/")
async def read_root():
    return {"message": "Welcome to the LangSAM API! - Prompt based Object Segmentation"}

@app.post("/segment")
async def segment_image(
    rgb_image: UploadFile = File(...),
    depth_image: UploadFile = File(None),
    prompt: str = Form(...)
):
    # Read and process RGB image
    rgb_content = await rgb_image.read()
    rgb_image = Image.open(io.BytesIO(rgb_content)).convert("RGB")
    
    # Process depth image if provided
    depth_array = None
    if depth_image:
        depth_content = await depth_image.read()
        depth_array = np.frombuffer(depth_content, dtype=np.uint16).reshape(rgb_image.size[::-1])

    # Perform segmentation
    masks, boxes, phrases, logits = model.predict(rgb_image, prompt)

    results = []
    for i, (mask, box) in enumerate(zip(masks, boxes)):
        mask_np = mask.cpu().numpy()
        center = get_segment_center(mask_np)
        
        result = {
            "phrase": phrases[i],
            "confidence": logits[i].item(),
            "bounding_box": box.tolist(),
            "center": center,
            "mask": mask_np.tolist()  # Convert to list for JSON serialization
        }
        
        if depth_array is not None:
            # Calculate average depth within the mask
            masked_depth = depth_array * mask_np
            avg_depth = np.mean(masked_depth[masked_depth > 0])
            result["average_depth"] = float(avg_depth)
        
        results.append(result)

    return JSONResponse(content={"results": results})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)