# Player Re-Identification in Sports Footage

This project performs real-time **Player Re-Identification (Re-ID)** in football match footage using a YOLOv8-based detection pipeline, DeepSORT tracking, and feature matching to consistently assign IDs to players across frames.

## 📌 Features

- Detects players, referees, balls using a fine-tuned YOLOv8 model (`best.pt`)
- Tracks objects using DeepSORT (Kalman Filter + Re-ID Embedding Matching)
- Re-identifies players using custom feature extractor
- Crops and saves per-ID images for post-analysis
- Visual overlay of bounding boxes and consistent tracking IDs


## 🧠 How it Works

1. **Detection**:
   - Uses YOLOv8 to detect all objects in a frame: players, referees, ball, etc.

2. **Tracking**:
   - DeepSORT takes detection boxes and confidences to assign consistent IDs across frames.
   - The detections are converted from (x1, y1, x2, y2) to (x_center, y_center, width, height).

3. **Re-Identification**:
   - Crops are extracted from each player bounding box.
   - A feature extractor generates embeddings for each player crop.
   - Cosine similarity is used to match new crops with existing ones.

4. **Display**:
   - Frame is rendered with bounding boxes and stable IDs using OpenCV.
   - Outputs can be saved for review.

## 🛠️ Requirements

- Python 3.10+
- Install required packages:

```bash
git clone https://github.com/aryan6336/player_re_identification
python -m venv venv
./venv/Scripts/activate
pip install -r requirements.txt
```
## download model best.pt 

```bash
python download_model.py
```
## final command to run the file 

```bash
python main.py
```
## 🔍 Customization

- Replace `best.pt` with your own trained YOLO model if needed.
- Edit `class_map` in `main.py` to match your class labels.

## 📦 Output

- Annotated video with consistent player IDs
- Folder of cropped images organized by track ID: `reid_data/{id}/`

## 🧪 Future Improvements

- Add jersey number OCR-based verification
- Integrate face-based Re-ID if visible
- Improve feature matching using Siamese Networks

## 📄 License

MIT License

## ✍️ Author

**Aryan Kumar** – [IIT Indore]
