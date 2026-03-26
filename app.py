import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile

# Page config
st.set_page_config(page_title="Object Tracking App", layout="wide")

st.title("⚽ Multi-Object Detection & Tracking")
st.markdown("Upload a video and track players using YOLOv8")

# Sidebar controls
st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.4)

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:

    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.video(uploaded_file)

    if st.button("🚀 Start Tracking"):

        st.write("Processing...")

        model = YOLO("yolov8s.pt")
        cap = cv2.VideoCapture(tfile.name)

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

            if results[0].boxes is not None:
                boxes = results[0].boxes
                coords = boxes.xyxy.cpu().numpy()
                classes = boxes.cls.cpu().numpy()
                ids = boxes.id

                if ids is not None:
                    ids = ids.cpu().numpy()

                for i, (box, cls) in enumerate(zip(coords, classes)):
                    if int(cls) != 0:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    track_id = int(ids[i]) if ids is not None else -1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {track_id}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0), 2)

            # Show frame in Streamlit
            frame_placeholder.image(frame, channels="BGR")

        cap.release()
        st.success("✅ Processing complete!")