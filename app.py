import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile

st.title("Human Detection with YOLOv8")

model = YOLO("yolov8n.pt")
PERSON_CLASS_ID = 0

def draw_boxes(frame, boxes):
    for box in boxes:
        cls = int(box.cls[0])
        if cls == PERSON_CLASS_ID:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Person {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

option = st.radio("Select Input:", ["Image", "Video"])

if option == "Image":
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        results = model(tfile.name)
        img = results[0].plot()
        st.image(img, caption="Detection Result", use_column_width=True)

elif option == "Video":
    file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, stream=True)
            for r in results:
                frame = draw_boxes(frame, r.boxes)
            stframe.image(frame, channels="BGR", use_column_width=True)
        cap.release()
