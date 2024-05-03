import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
import mediapipe as mp

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(torch.__version__)

# Config values
# video_path = "data_ext/people.mp4"
video_path = 0
conf_threshold = 0.1
tracking_class = 0 # None: track all

# Initialize DeepSort
tracker = DeepSort(max_age=20)

# Initialize YOLOv9
model  = DetectMultiBackend(weights="weights/yolov9-c-converted.pt", device=device, fuse=True)
model  = AutoShape(model)

# Load class names from file
with open("data_ext/classes.names") as f:
    class_names = f.read().strip().split('\n')

colors = np.random.randint(0,255, size=(len(class_names),3))
tracks = []

# Initialize VideoCapture to read from video file
cap = cv2.VideoCapture(video_path)

# Initialize MediaPipe Pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Detect objects using YOLOv9
    results = model(frame)

    # Extract bounding boxes and class IDs
    detect = []
    for detect_object in results.pred[0]:
        label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)

        if tracking_class is None:
            if confidence < conf_threshold:
                continue
        else:
            if class_id != tracking_class or confidence < conf_threshold:
                continue

        detect.append([[x1, y1, x2-x1, y2 - y1], confidence, class_id])

    # Update tracks using DeepSort
    tracks = tracker.update_tracks(detect, frame=frame)

    # Draw bounding boxes and IDs
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id

            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            color = colors[class_id]
            B, G, R = map(int, color)

            label = "{}-{}".format(class_names[class_id], track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Use MediaPipe to detect human poses for each detected person
            image = frame[y1:y2, x1:x2]
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame[y1:y2, x1:x2], results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

     # Show hình ảnh lên màn hình
    # desired_width = 500  # Độ rộng mong muốn của cửa sổ hiển thị
    # desired_height =800
    # # scale_factor = desired_width / frame.shape[1]  # Tính tỷ lệ scale dựa trên độ rộng mong muốn
    # # desired_height = int(frame.shape[0] * scale_factor)  # Tính độ cao tương ứng

    # # Resize frame
    # resized_frame = cv2.resize(frame, (desired_width, desired_height))

    # # Hiển thị hình ảnh đã resize
    # cv2.imshow("DemoDtAIRC-H3", resized_frame)
    cv2.imshow("DemoDtAIRC-H3", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
