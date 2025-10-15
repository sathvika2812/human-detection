import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import time

st.set_page_config(page_title="Human Detection YOLOv8", layout="wide")
st.title("👀 Human Detection YOLOv8 (Headless)")

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
PERSON_CLASS_ID = 0

# Create output folder
os.makedirs("output", exist_ok=True)


def draw_boxes(frame, boxes):
    """Draw bounding boxes for detected humans."""
    for box in boxes:
        cls = int(box.cls[0])
        if cls == PERSON_CLASS_ID:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Person {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def find_working_camera(max_index=5):
    """Automatically find working webcam index."""
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
        cap.release()
    return None


def process_video(source):
    """Process video file, webcam, or RTSP."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error(f"Cannot open video source: {source}")
        return

    # Prepare output video writer
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_file = os.path.join("output", f"output_{int(time.time())}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    stframe = st.empty()
    prev_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 inference
        results = model(frame, stream=True)
        for r in results:
            frame = draw_boxes(frame, r.boxes)

        # FPS overlay
        curr_time = time.time()
        fps_text = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {fps_text:.2f}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        out.write(frame)
        stframe.image(frame, channels="BGR", use_column_width=True)

    cap.release()
    out.release()
    st.success(f"Output saved: {output_file}")


option = st.radio("Select Input:", ["Image", "Video", "Webcam", "RTSP"])

if option == "Image":
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        results = model(tfile.name)
        img = results[0].plot()
        st.image(img, caption="Detection Result", use_column_width=True)
        output_path = os.path.join("output", f"image_{int(time.time())}.jpg")
        cv2.imwrite(output_path, img)
        st.success(f"Saved output image: {output_path}")

elif option == "Video":
    file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        process_video(tfile.name)

elif option == "Webcam":
    cam_index = find_working_camera()
    if cam_index is None:
        st.error("No webcam found.")
    else:
        st.info(f"Using webcam index {cam_index}")
        process_video(cam_index)

elif option == "RTSP":
    rtsp_url = st.text_input("Enter RTSP URL")
    if rtsp_url:
        process_video(rtsp_url)
