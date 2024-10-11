import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image
from lang_sam import LangSAM
import threading
import queue

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

def prompt_input_thread(prompt_queue):
    while True:
        text_prompts = input("Enter objects to detect (comma-separated) or 'q' to quit: ")
        prompt_queue.put(text_prompts)
        if text_prompts.lower() == 'q':
            break

def main():
    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # Initialize LangSAM model
    model = LangSAM()

    # Initialize prompt queue and current prompts
    prompt_queue = queue.Queue()
    current_prompts = []

    # Start input thread
    input_thread = threading.Thread(target=prompt_input_thread, args=(prompt_queue,))
    input_thread.daemon = True
    input_thread.start()

    try:
        while True:
            # Check for new prompts
            try:
                new_prompts = prompt_queue.get_nowait()
                if new_prompts.lower() == 'q':
                    break
                current_prompts = [prompt.strip() for prompt in new_prompts.split(',')]
                print(f"Updated prompts: {current_prompts}")
            except queue.Empty:
                pass

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            # Convert to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

            # Perform segmentation for each prompt
            for prompt in current_prompts:
                masks, boxes, phrases, logits = model.predict(pil_image, prompt)

                # Process results
                for i, (mask, box) in enumerate(zip(masks, boxes)):
                    # Convert mask to numpy array
                    mask_np = mask.cpu().numpy()

                    # Apply mask to image
                    color_image[mask_np] = color_image[mask_np] * 0.5 + np.array([0, 0, 255]) * 0.5

                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Get and draw segment center
                    center = get_segment_center(mask_np)
                    if center:
                        cv2.circle(color_image, center, 5, (255, 0, 0), -1)
                        cv2.putText(color_image, f"Center: {center}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Display phrase and confidence
                    phrase = phrases[i]
                    confidence = logits[i].item()
                    cv2.putText(color_image, f"{phrase}: {confidence:.2f}", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display current prompts on the image
            prompt_text = ", ".join(current_prompts)
            cv2.putText(color_image, f"Prompts: {prompt_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show images
            cv2.imshow('RealSense LangSAM Segmentation', color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()