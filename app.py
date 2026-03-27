import streamlit as st
import cv2
from ultralytics import YOLO

st.set_page_config(page_title="Object Tracking App", layout="wide")
st.title("⚽ Multi-Object Detection & Tracking")

VIDEO_PATH = "foot_ball.mp4"

st.video(VIDEO_PATH)

if st.button("🚀 Start Tracking"):

    model = YOLO("yolov8s.pt")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        st.error("❌ Cannot open video file")
        st.stop()

    frame_placeholder = st.empty()
    track_history = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        try:
            frame = cv2.resize(frame, (800, 450))
        except Exception as e:
            st.warning(f"Resize error: {e}")
            continue

        try:
            results = model.track(
                frame,
                persist=True,
                conf=0.3,
                iou=0.5,
                tracker="bytetrack.yaml"
            )
        except Exception as e:
            st.warning(f"Tracking error: {e}")
            continue

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            coords = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            ids = boxes.id

            if ids is not None:
                ids = ids.cpu().numpy()

                for i in range(min(len(coords), len(classes), len(ids))):
                    cls = int(classes[i])
                    if cls != 0:
                        continue

                    x1, y1, x2, y2 = map(int, coords[i])
                    track_id = int(ids[i])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {track_id}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0), 2)

                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if track_id not in track_history:
                        track_history[track_id] = []
                    track_history[track_id].append((cx, cy))
                    if len(track_history[track_id]) > 30:
                        track_history[track_id].pop(0)

                    for j in range(1, len(track_history[track_id])):
                        cv2.line(frame,
                                 track_history[track_id][j - 1],
                                 track_history[track_id][j],
                                 (0, 255, 0), 2)

        frame_placeholder.image(frame, channels="BGR")

    cap.release()
    st.success("✅ Tracking complete!")
