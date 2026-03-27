import streamlit as st
import cv2
import os
from ultralytics import YOLO

st.set_page_config(page_title="Object Tracking App", layout="wide")
st.title("⚽ Multi-Object Detection & Tracking")

VIDEO_PATH = "foot_ball.mp4"
OUTPUT_PATH = "output_tracked.mp4"

st.video(VIDEO_PATH)

col1, col2, col3 = st.columns(3)
conf_thresh  = col1.slider("Confidence Threshold", 0.1, 0.9, 0.2, 0.05)
iou_thresh   = col2.slider("IOU Threshold",        0.1, 0.9, 0.4, 0.05)
trail_length = col3.slider("Trail Length",         10,  60,  30,  5)

if st.button("🚀 Start Tracking"):

    model = YOLO("yolov8s.pt")
    cap   = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        st.error("❌ Cannot open video file")
        st.stop()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    OUT_W, OUT_H = 960, 540

    raw_path = "raw_tracked.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(raw_path, fourcc, fps, (OUT_W, OUT_H))

    progress_bar  = st.progress(0)
    status_text   = st.empty()
    track_history = {}
    frame_count   = 0

    with st.spinner("⏳ Processing video, please wait..."):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_count += 1
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            status_text.text(f"Processing frame {frame_count} / {total_frames}")

            try:
                frame = cv2.resize(frame, (OUT_W, OUT_H))
            except Exception:
                continue

            try:
                results = model.track(
                    frame,
                    persist=True,
                    conf=conf_thresh,
                    iou=iou_thresh,
                    tracker="bytetrack.yaml",
                    verbose=False
                )
            except Exception:
                writer.write(frame)
                continue

            if results and results[0].boxes is not None:
                boxes   = results[0].boxes
                coords  = boxes.xyxy.cpu().numpy()
                classes = boxes.cls.cpu().numpy()
                ids     = boxes.id

                if ids is not None:
                    ids = ids.cpu().numpy()
                    for i in range(min(len(coords), len(classes), len(ids))):
                        if int(classes[i]) != 0:
                            continue
                        x1, y1, x2, y2 = map(int, coords[i])
                        track_id        = int(ids[i])

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label       = f"ID {track_id}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
                        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        if track_id not in track_history:
                            track_history[track_id] = []
                        track_history[track_id].append((cx, cy))
                        if len(track_history[track_id]) > trail_length:
                            track_history[track_id].pop(0)
                        pts = track_history[track_id]
                        for j in range(1, len(pts)):
                            alpha = int(255 * j / len(pts))
                            cv2.line(frame, pts[j - 1], pts[j], (0, alpha, 0), 2)

            cv2.putText(frame, f"Frame: {frame_count}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            writer.write(frame)

    cap.release()
    writer.release()

    status_text.text("🔄 Re-encoding to H.264 for browser playback...")
    ret = os.system(f"ffmpeg -y -i {raw_path} -vcodec libx264 -pix_fmt yuv420p -crf 23 -preset fast {OUTPUT_PATH}")

    if ret != 0 or not os.path.exists(OUTPUT_PATH) or os.path.getsize(OUTPUT_PATH) == 0:
        OUTPUT_PATH_FINAL = raw_path
    else:
        OUTPUT_PATH_FINAL = OUTPUT_PATH

    progress_bar.progress(1.0)
    status_text.text(f"✅ Done! Processed {frame_count} frames.")
    st.success("✅ Tracking complete! Download or watch below 👇")

    with open(OUTPUT_PATH_FINAL, "rb") as f:
        video_bytes = f.read()

    st.download_button(
        label="⬇️ Download Tracked Video",
        data=video_bytes,
        file_name="tracked_output.mp4",
        mime="video/mp4"
    )

    st.video(OUTPUT_PATH_FINAL)