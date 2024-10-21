import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from torchvision.transforms import Resize
from PIL import Image
import clip

# Load SAM
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# Load CLIP
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def get_sam_input_points(text, image):
    # Encode image and text with CLIP
    image_input = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
    text_input = clip.tokenize([text]).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_input)

    # Compute similarity
    similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
    
    # Get the most similar region
    similarity = similarity.reshape(1, 1, 7, 7)
    similarity = Resize((image.shape[0], image.shape[1]))(similarity).squeeze()
    
    # Get the point with maximum similarity
    max_idx = similarity.argmax()
    x = max_idx % image.shape[1]
    y = max_idx // image.shape[1]
    
    return np.array([[x, y]])

# Initialize video capture
cap = cv2.VideoCapture(2)  # Use 0 for webcam, or provide video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Set the image for SAM
    predictor.set_image(frame)
    
    # Get user input
    text_prompt = input("Enter object to segment (or 'q' to quit): ")
    if text_prompt.lower() == 'q':
        break
    
    # Get input point for SAM
    input_point = get_sam_input_points(text_prompt, frame)
    
    # Generate mask
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=np.array([1]),
        multimask_output=False,
    )
    
    # Apply mask to frame
    mask = masks[0]
    frame[mask] = frame[mask] * 0.5 + np.array([0, 0, 255]) * 0.5
    
    # Display result
    cv2.imshow('Segmentation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()