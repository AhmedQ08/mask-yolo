import torch
from ultralytics import YOLO
import cv2

# Step 1: Load the YOLOv8 model
model = YOLO('best (3).pt')  # Load your YOLOv8 model (replace with your model path)

# Step 2: Access the webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam; change if using an external camera

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Step 3: Process video frames
while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break

    # Run inference
    results = model(frame)

    # Draw bounding boxes and labels on the frame
    annotated_frame = results[0].plot()  # Annotate the frame with bounding boxes and labels

    # Display the frame with detections
    cv2.imshow('Mask Detection', annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
