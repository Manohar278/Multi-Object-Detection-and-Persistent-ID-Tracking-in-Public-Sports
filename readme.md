# ⚽ Multi-Object Detection and Persistent ID Tracking in Public Sports

A real-time football player detection and tracking pipeline built with YOLOv8 and ByteTrack,
deployed as an interactive Streamlit web app.


---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| streamlit | >=1.35.0 | Web app framework |
| ultralytics | >=8.2.0 | YOLOv8 model and ByteTrack tracker |
| torch | >=2.2.0 | Deep learning backend |
| numpy | >=1.26.0 | Numerical operations |
| lapx | >=0.5.5 | Linear assignment solver for ByteTrack |
| python3-opencv | apt | Video read/write and frame annotation |
| ffmpeg | apt | Re-encodes output video to H.264 |
| libgl1 | apt | OpenCV system dependency |

---


```

### 2. Install system packages (Linux/Ubuntu)
```bash
sudo apt-get update
sudo apt-get install -y python3-opencv libgl1 ffmpeg
```


```

### 5. Place required files in the project root
```
├── app.py
├── foot_ball.mp4        ← input video
├── yolov8s.pt           ← YOLOv8s pretrained weights
├── requirements.txt
├── packages.txt
├── runtime.txt
└── README.md
```

---

## ▶️ How to Run the Pipeline

### Run locally
```bash
streamlit run app.py
```
Open browser at `http://localhost:8501`

### Run on Streamlit Cloud
1. Push all files to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set `app.py` as the entry point
4. Deploy — Streamlit Cloud auto-installs `packages.txt` and `requirements.txt`

### Using the app
1. The original video previews at the top
2. Adjust sliders:
   - **Confidence Threshold** — minimum detection score (lower = more detections)
   - **IOU Threshold** — overlap threshold for suppressing duplicate boxes
   - **Trail Length** — how many past positions to draw per player
3. Click **🚀 Start Tracking**
4. Wait for processing and re-encoding to complete
5. Output tracked video plays automatically in the browser

---

## 💡 Assumptions

- Input video is named `foot_ball.mp4` and placed in the project root
- YOLOv8s weights file `yolov8s.pt` is placed in the project root
- Only **person** detections (COCO class `0`) are tracked
- ByteTrack config `bytetrack.yaml` is auto-downloaded by Ultralytics on first run
- `ffmpeg` is available on the system (installed via `packages.txt` on Streamlit Cloud)
- Video is a standard football broadcast with a relatively stable camera angle

---

## ⚠️ Limitations

- **Processing speed** — inference runs on CPU on Streamlit Cloud; a 30-second
  video may take several minutes to process
- **Small/distant players** — YOLOv8s may miss players far from the camera or
  partially occluded by others
- **ID switching** — ByteTrack can reassign IDs when players overlap for extended
  periods or move off-screen and return
- **Crowd detections** — at low confidence thresholds the model sometimes detects
  spectators in the stands as persons
- **Single video only** — the app is hardcoded to one input video with no upload interface
- **No GPU** — Streamlit Cloud free tier provides CPU only
- **Memory** — very long videos may exhaust available RAM when loading output bytes

---

## 🤖 Model and Tracker Choice

### YOLOv8s
- Single-stage anchor-free detector pretrained on COCO
- Detects persons (class 0) out of the box with no fine-tuning
- Small variant gives the best speed/accuracy tradeoff for CPU deployment
- Unified `model.track()` API simplifies the pipeline

### ByteTrack
- Two-stage IoU-based tracker — no ReID model needed
- Recovers low-confidence detections that SORT would discard
- Built into Ultralytics via `tracker="bytetrack.yaml"`
- Competitive MOTA/HOTA scores on MOT benchmarks with fewer ID switches than SORT

---

## 📁 File Structure
```
├── app.py                  ← Streamlit application
├── foot_ball.mp4           ← Input video
├── yolov8s.pt              ← YOLOv8s weights
├── requirements.txt        ← Python dependencies
├── packages.txt            ← System apt dependencies
├── runtime.txt             ← Python version for Streamlit Cloud
└── README.md               ← This file
```