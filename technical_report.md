# Technical Report: Multi-Object Detection and Persistent ID Tracking

**Assignment:** Multi-Object Detection and Persistent ID Tracking in Public Sports Footage  
 
**Pipeline:** YOLOv8s + ByteTrack | Deployed via Streamlit  

---

## 1. Model / Detector Used

### YOLOv8s — You Only Look Once Version 8 (Small)

The detection backbone is **YOLOv8s** by Ultralytics, pretrained on the **COCO dataset**
(80 classes, person = class index 0).

YOLOv8 is a single-stage anchor-free detector that performs classification and bounding
box regression in one forward pass, making it significantly faster than two-stage
detectors like Faster R-CNN.

Key properties:
- **Anchor-free head** — predicts object centers directly
- **C2f bottleneck blocks** — improved gradient flow over YOLOv5
- **Decoupled head** — separate branches for classification and regression
- **Input resolution** — frames resized to 960×540 before inference
- Only **class 0 (person)** detections are retained; all others are discarded

---

## 2. Tracking Algorithm Used

### ByteTrack (Zhang et al., 2022)

ByteTrack is integrated natively through Ultralytics via `tracker="bytetrack.yaml"`
and `persist=True`.

**How it works:**

**Stage 1 — High-confidence association**
Detections above a confidence threshold are matched to existing tracklets using the
Hungarian algorithm with IoU as the cost metric.

**Stage 2 — Low-confidence recovery**
Detections below the primary threshold (discarded by other trackers) are used in a
second association pass against unmatched tracklets, recovering players that are
briefly occluded or detected with lower confidence.

A **Kalman filter** models each player's velocity and predicts their next bounding box
position, enabling correct association even during fast movement. Unmatched tracklets
are held in a buffer before permanent deletion, allowing re-entry with the same ID.

---

## 3. Why This Combination Was Selected

| Criterion | Reasoning |
|---|---|
| No fine-tuning needed | YOLOv8s detects persons reliably on COCO weights |
| CPU deployable | Fast enough for offline processing without GPU |
| No ReID model | ByteTrack uses only IoU — no appearance network needed |
| Low-confidence recovery | Critical for sports footage with frequent occlusions |
| Native integration | Ultralytics `model.track()` unifies detection and tracking |
| Proven benchmarks | Competitive MOTA/HOTA on MOT17 and MOT20 |

**Alternatives considered and rejected:**

- **DeepSORT** — requires a separate ReID model, slower on CPU, marginal gain
- **SORT** — discards low-confidence detections, more ID switches in crowds
- **StrongSORT** — more accurate but 3-4x slower, not viable for CPU
- **YOLOv8m/l** — better accuracy but 2-4x slower inference on CPU

---

## 4. How ID Consistency Is Maintained

**a) Persistent tracking state (`persist=True`)**
The tracker maintains internal tracklet state across frames rather than resetting
each frame.

**b) Kalman Filter motion prediction**
Each tracklet models player velocity, predicting the next bounding box even when
the detector temporarily misses a player.

**c) Two-stage association**
Low-confidence detections are used in a second matching pass, recovering tracklets
that would otherwise be lost during brief occlusions.

**d) Tracklet buffer**
Unmatched tracklets are kept in a lost state for several frames before deletion,
allowing a player to re-enter the frame with the same ID.

**e) Trail visualization**
A 30-frame centroid history is drawn per player with a fading green trail, making
ID continuity visually verifiable in the output video.

---

## 5. Challenges Faced

**a) No GPU on deployment**
Streamlit Cloud provides CPU-only compute. YOLOv8s takes ~200–250ms per frame on
CPU, meaning a 30-second video at 30fps takes approximately 3–4 minutes to process.

**b) Crowd false positives**
The football stadium contains thousands of spectators. At low confidence thresholds,
front-row spectators are detected as persons and assigned tracking IDs.

**c) Browser video playback**
OpenCV's default `mp4v` codec is not browser-playable. An additional `ffmpeg`
re-encoding step was required with `libx264`, `yuv420p` pixel format, and
`-movflags +faststart` to make the video seekable in-browser.

**d) Python 3.14 compatibility**
The deployment platform used Python 3.14, for which `opencv-python-headless` has
no PyPI wheel. Resolved by installing `python3-opencv` via `apt` in `packages.txt`.

**e) Missing `lap` module**
ByteTrack requires a linear assignment solver (`lap`) which has no Python 3.14 wheel.
Resolved by substituting `lapx`, a maintained fork with modern Python support.

**f) Similar appearance confusion**
Players wearing the same jersey are visually nearly identical. Since ByteTrack uses
only IoU (no appearance features), it cannot distinguish between nearby players with
similar motion, occasionally swapping IDs during close contact.

---

## 6. Failure Cases Observed

| Failure Case | Description |
|---|---|
| ID switch on contact | When two players challenge for the ball and bounding boxes fully overlap, IDs occasionally swap after separation |
| Spectator detections | At confidence=0.20, some front-row spectators are detected and assigned tracking IDs |
| Missed far-field players | Players near the far end of the pitch (~20×40px) are intermittently missed, causing track fragmentation |
| Goalkeeper occlusion | Goalkeeper near the post is sometimes occluded by the net, causing track loss and re-entry with a new ID |
| Motion blur | Fast sprinting players introduce blur that reduces detection confidence below threshold |

---

## 7. Possible Improvements

**Detection:**
- Fine-tune YOLOv8m on a sports-specific dataset (SoccerNet, SportsMOT) for better
  detection of small and distant players
- Apply test-time augmentation (TTA) to improve recall for partially visible players
- Add a ROI mask to exclude the spectator area and eliminate crowd false positives

**Tracking:**
- Replace ByteTrack with **BoT-SORT** or **StrongSORT** which incorporate ReID
  appearance embeddings for better identity preservation through occlusions
- Add a jersey color classifier as a secondary appearance cue to distinguish teams

**Pipeline:**
- Implement frame skipping with bounding box interpolation to increase throughput
- Add speed estimation by converting pixel displacement to real-world distance via
  homography using known pitch dimensions
- Add a file uploader in the Streamlit app for any input video

**Evaluation:**
- Compute MOTA, HOTA, and IDF1 metrics against manually annotated ground truth

---

## 8. Optional Enhancements Implemented

- **Trail visualization** — 30-frame centroid history drawn as a fading green trail
  showing movement direction and speed
- **Live sliders** — confidence, IOU, and trail length tunable without redeploying
- **Progress bar and frame counter** — real-time processing feedback
- **Streamlit deployment** — fully browser-accessible pipeline with no local setup

---

## 9. Pipeline Summary
```
Input Video (foot_ball.mp4)
        │
        ▼
  Frame Extraction (OpenCV)
        │
        ▼
  Resize to 960×540
        │
        ▼
  YOLOv8s Inference
  (person class only, conf=0.20, iou=0.40)
        │
        ▼
  ByteTrack Association
  (Kalman filter + two-stage IoU matching)
        │
        ▼
  Bounding Box + ID Annotation
  + Trail Drawing (30-frame history)
        │
        ▼
  Write Frame to VideoWriter (mp4v)
        │
        ▼
  ffmpeg Re-encode (H.264 + faststart)
        │
        ▼
  Output Video (browser-playable MP4)
