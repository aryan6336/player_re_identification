
# 🎯 Player Re-Identification in Sports Footage  
### 🚀 Final Report

## 📁 Project Overview
This project focuses on re-identifying players in sports footage using computer vision techniques. A YOLOv8 object detection model is combined with DeepSORT tracking and enhanced by face-based feature extraction for robust identity preservation even during occlusions or missed detections.

---

## 🛠️ Tech Stack
- **YOLOv8**: Person, referee, and ball detection
- **DeepSORT**: Real-time object tracking
- **Torchreid**: Feature extraction for re-identification
- **OpenCV**: Image processing
- **Python**: Main implementation
- **CUDA/GPU (Optional)**: For faster inference

---

## 🧠 Key Modules

### `main.py`
- Initializes YOLO model
- Captures frames from video
- Extracts bounding boxes and confidences
- Passes detections to custom `tracker.py`
- Draws results with consistent IDs

### `tracker.py`
- Converts bounding boxes to DeepSORT format
- Extracts face crops from tracked players
- Embeds face features via Torchreid's extractor
- Matches new features against known player embeddings
- Re-uses previous IDs to achieve player re-identification

---

## 🧪 Workflow

1. **Detection**  
   YOLOv8 detects:
   - Players (class_id = 2)
   - Goalkeepers (class_id = 1)
   - Referees (class_id = 3)
   - Ball (class_id = 0)

2. **Tracking**  
   DeepSORT maintains identity across frames using Kalman Filter and cosine distance.

3. **Feature Extraction**  
   Cropped face patches are passed to the `FeatureExtractor` (from `torchreid`) to generate embeddings.

4. **Re-identification**  
   - Feature embeddings are matched with stored vectors using cosine similarity.
   - If matched, the old ID is used.
   - If not matched, a new ID is assigned.

---

## 🧪 Sample Output

For each frame:
```
ID: 6 | Coordinates: [x1, y1, x2, y2]
ID: 2 | Coordinates: [x1, y1, x2, y2]
...
```

---

## ⚠️ Challenges Faced

- 🧾 **Import Errors**: Misalignment in Torchreid path (`from torchreid.reid.utils.feature_extractor`).
- 📐 **Shape Errors**: Empty crops led to feature extraction errors.
- 🔁 **Function Argument Mismatch**: Fixed `update()` call mismatch by adjusting arguments in `main.py`.
- 🔍 **Face not visible**: In some frames, lack of clear face view affects re-identification.

---

## 📊 Performance Tips

- Use **GPU acceleration** for faster frame processing.
- Ensure **face visibility** in camera feed for higher re-ID accuracy.
- Tune DeepSORT parameters like `max_age`, `n_init`, and `nms_max_overlap`.

---

## 🧩 Future Improvements

- Integrate **jersey number OCR** as a secondary re-identification strategy.
- Use **clip-based embeddings** or **multi-modal features**.
- Add support for **multi-camera** views with homography stitching.

---

## 👨‍💻 Author

Aryan Kumar  
Player Re-Identification System  
Internship Task Submission – June 2025
