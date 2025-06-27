import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from featureExtraction import FeatureExtractor

# Initialize model
model = YOLO("models/best.pt")  # Use 'yolov8n' for speed
model.fuse()

# Initialize feature extractor
extractor = FeatureExtractor()

# Initialize DeepSORT
deepsort = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

# Video input
cap = cv2.VideoCapture(0)  # use webcam; replace with "video.mp4" if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    detections = []
    crops = []
    bboxes = []
    confidences = []

    for result in results.boxes:
        cls_id = int(result.cls[0])
        conf = float(result.conf[0])
        if cls_id == 2:  # Class ID for 'person' in COCO
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            crops.append(frame[y1:y2, x1:x2])
            bboxes.append([x1, y1, x2 - x1, y2 - y1])  # (x, y, w, h)
            confidences.append(conf)

    if crops:
        features = extractor.extract(crops)
        for bbox, conf, feat in zip(bboxes, confidences, features):
            detections.append((bbox, conf, feat))

        tracks = deepsort.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, w, h = map(int, track.to_ltrb())
            cv2.rectangle(frame, (l, t), (l + w, t + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Real-Time Re-ID", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


