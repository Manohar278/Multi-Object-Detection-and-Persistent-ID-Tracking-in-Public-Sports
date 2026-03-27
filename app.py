import streamlit as st
import cv2
from ultralytics import YOLO

st.set_page_config(page_title="Object Tracking App", layout="wide")
st.title("⚽ Multi-Object Detection & Tracking")

VIDEO_PATH = "foot ball.mp4"  # no spaces!

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
        if not ret:
            break

        frame = cv2.resize(frame, (800, 450))

        results = model.track(
            frame,
            persist=True,
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
                if int(classes[i]) != 0:
                    continue

                x1, y1, x2, y2 = map(int, coords[i])
                track_id = int(ids[i]) if ids is not None else -1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0), 2)

        frame_placeholder.image(frame, channels="BGR")

    cap.release()
    st.success("✅ Done!")