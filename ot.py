import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Allowed object classes to track
allowed_classes = ["person", "car", "truck", "bus", "motorcycle", "bicycle"]

# Load video
video_path = "input_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Failed to open input video!")
    exit()

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0.0:
    fps = 30

# Output writer
out = cv2.VideoWriter("output_filtered.mp4", cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))
if not out.isOpened():
    print("❌ Failed to open video writer!")
    exit()

# Process video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, verbose=False)
    annotated_frame = frame.copy()

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            if class_name in allowed_classes:
                # Bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]

                label = f"{class_name} {conf:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(annotated_frame)
    cv2.imshow("Filtered Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Video processing complete. Output saved as 'output_filtered.mp4'.")
