# Trafic Lights detections

# İmport libraries
import cv2
import supervision as sv
from ultralytics import YOLO

# Trained model
model = YOLO('bestv8.pt')

# Video Source
cap = cv2.VideoCapture('video2.mp4')

# İf video not open
if not cap.isOpened():
    print("Kameradan görüntü gelmiyor...")

# Annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

frame_id = 0


while True:
    # Read video
    ret, frame = cap.read()

    if not ret:
        break
    
    # Blur
    frame = cv2.medianBlur(frame,3)

    # Operate every 3 frames
    if frame_id % 3 == 0:

        # If you want to increase accuracy, change the "conf" value
        results = model(frame,conf=0.05,agnostic_nms = False)[0]
        detections = sv.Detections.from_ultralytics(results)

        annotated_image = bounding_box_annotator.annotate(
            scene=frame, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections)
    
    frame_id += 1

    cv2.imshow("img", annotated_image)

    # İf you want quit press "q"
    if cv2.waitKey(1)& 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
