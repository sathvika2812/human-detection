import cv2
import os
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
PERSON_CLASS_ID = 0  # COCO class for 'person'

# Create output folder if it doesn't exist
os.makedirs("output", exist_ok=True)


def find_working_camera(max_index=5):
    """Automatically find a working webcam index."""
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
        cap.release()
    return None


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


def detect_image(image_path):
    results = model(image_path)
    for r in results:
        img = r.orig_img.copy()
        img = draw_boxes(img, r.boxes)
        output_path = os.path.join("output", os.path.basename(image_path))
        cv2.imwrite(output_path, img)
        print(f"Saved output image at {output_path}")


def detect_video(source):
    """Video, webcam, or RTSP feed."""
    # Auto-find webcam if source == 0
    if source == 0:
        cam_index = find_working_camera()
        if cam_index is None:
            raise Exception("No working webcam found.")
        source = cam_index

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise Exception(f"Cannot open source: {source}")

    # Prepare output video writer
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_file = os.path.join("output", "output_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    prev_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True)
        for r in results:
            frame = draw_boxes(frame, r.boxes)

        # FPS display
        curr_time = time.time()
        fps_text = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {fps_text:.2f}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        out.write(frame)  # save to output
        cv2.imshow("Human Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved output video at {output_file}")


if __name__ == "__main__":
    print("""
    Select Input Type:
    1 - Image
    2 - Video
    3 - Webcam
    4 - RTSP Stream
    """)

    choice = input("Enter choice (1-4): ")

    if choice == '1':
        image_path = input("Enter image path: ")
        detect_image(image_path)

    elif choice == '2':
        video_path = input("Enter video path: ")
        detect_video(video_path)

    elif choice == '3':
        print("Opening webcam...")
        detect_video(0)

    elif choice == '4':
        rtsp_url = input("Enter RTSP stream URL: ")
        detect_video(rtsp_url)

    else:
        print("Invalid choice. Please select 1–4.")
