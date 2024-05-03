import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
import threading
import playsound
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(torch.__version__)

# Config values
video_path = "data_ext/haland.mp4"
conf_threshold = 0.1
tracking_class = None  # None: track all
skip_frames = 1  # Process every 3rd frame
frame_count = 0

# Initialize DeepSort
tracker = DeepSort(max_age=3)

# Initialize YOLOv9
model = DetectMultiBackend(weights="weights/best-4acts-v2.pt", device=device, fuse=True)
model = AutoShape(model)

# Load class names
with open("data_ext/classes-action.names") as f:
    class_names = f.read().strip().split('\n')

colors = np.random.randint(0, 255, size=(len(class_names), 3))
tracks = []

# Initialize VideoCapture to read from the video file
cap = cv2.VideoCapture(video_path)

# Set to keep track of appeared track IDs
appeared_ids = set()
people_count = 0

# Load alert sound
alert_sound_path = "sound/sound_warning_1 (mp3cut.net).wav"

# Function to play alert sound
def play_alert_sound():
    global frame
    playsound.playsound(alert_sound_path)

    # Flash red border around the frame for 3 seconds
    start_time = time.time()
    while time.time() - start_time < 3:
        # Draw red border around the frame
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
        cv2.imshow("DemoDtAIRC-H3", frame)
        cv2.waitKey(25)  # Wait for 25 milliseconds
        # Clear the red border by drawing a black rectangle over it
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), 10)
        cv2.imshow("DemoDtAIRC-H3", frame)
        cv2.waitKey(25)  # Wait for 25 milliseconds

# Function to select object using mouse click
selected_object_id = None
def select_object(event, x, y, flags, param):
    global selected_object_id

    if event == cv2.EVENT_LBUTTONDOWN:
        # Loop through tracks to find if the click is inside any bounding box
        for track in tracks:
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_object_id = track.track_id
                print("Selected object ID:", selected_object_id)
                break

# Register mouse callback function
cv2.namedWindow("DemoDtAIRC-H3")
cv2.setMouseCallback("DemoDtAIRC-H3", select_object)

# Read frames from the video
while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % skip_frames != 0:
        continue
    
    # Detect objects using the model
    results = model(frame)

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

        detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

    # Update tracks using DeepSort
    checkclass=0
    num_persons = sum(1 for obj in detect if obj[2] == checkclass)
    tracks = tracker.update_tracks(detect, frame=frame)

    # Draw bounding boxes and IDs on the frame
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id

            # Only count if the track ID has not appeared before
            if track_id not in appeared_ids:
                appeared_ids.add(track_id)
                if class_names[class_id] == "Falling" or class_names[class_id] == "Fighting":
                    # Create a new thread to play alert sound
                    alert_thread = threading.Thread(target=play_alert_sound)
                    alert_thread.start()

                    # Draw red alert on the frame
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, ltrb)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Alert: " + class_names[class_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Get coordinates and class_id to draw on the image
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            color = colors[class_id]
            B, G, R = map(int, color)
            confidence = float(confidence)
            confidence = round(confidence, 2)
            label = "{}-{}-{}".format(class_names[class_id], track_id, confidence)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 1)
            cv2.rectangle(frame, (x1 , y1 - 20), (x1 + len(label) * 8, y1), (B, G, R), -1)    #148,238,148
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255), 1)

            # If a specific object is selected, draw bounding box for it with a different color
            if selected_object_id is not None and track_id == selected_object_id:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
    num_persons_text = f"Number prs: {num_persons}"
    cv2.putText(frame, num_persons_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # Show frame
    cv2.imshow("DemoDtAIRC-H3", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()