import streamlit as st
import cv2
from ultralytics import YOLO

# Page config
st.set_page_config(page_title="Object Tracking App", layout="wide")

st.title("⚽ Multi-Object Detection & Tracking")
st.markdown("Tracking players using YOLOv8 (preloaded video)")

# Sidebar controls
st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.4)

# Predefined video path
VIDEO_PATH = "football.mp4"   # make sure this file exists

# Show video preview
st.video(VIDEO_PATH)

if st.button("🚀 Start Tracking"):

    st.write("Processing...")

    # Load model
    model = YOLO("yolov8s.pt")

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        st.error("❌ Cannot open video file")
        st.stop()

    frame_placeholder = st.empty()
    track_history = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (800, 450))

        results = model.track(
            frame,
            persist=True,
            conf=confidence,
            tracker="bytetrack.yaml"
        )

        if results and results[0].boxes is not None:

            boxes = results[0].boxes
            coords = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            ids = boxes.id

            if ids is not None:
                ids = ids.cpu().numpy()

            for i in range(len(coords)):

                if int(classes[i]) != 0:  # only person
                    continue

                x1, y1, x2, y2 = map(int, coords[i])
                track_id = int(ids[i]) if ids is not None else -1

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0), 2)

                # Track path
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
                             (0, 255, 0),
                             2)

        # Display frame
        frame_placeholder.image(frame, channels="BGR")

    cap.release()
    st.success("✅ Processing complete!")