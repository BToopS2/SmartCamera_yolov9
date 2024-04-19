#HHH
# import cv2
# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.is_available())
# print(torch.__version__)
# import numpy as np
# from deep_sort_realtime.deepsort_tracker import DeepSort
# from models.common import DetectMultiBackend, AutoShape
# # from ultralytics import YOLO 

# # Config value
# # video_path = "data_ext/test4.mp4"
# video_path="rtsp://admin:Qazxsw123@192.168.88.20:554/cam/realmonitor?channel=1&subtype=0"
# # video_path=0
# conf_threshold = 0.1
# tracking_class = 0 # None: track all

# # Khởi tạo DeepSort
# tracker = DeepSort(max_age=20)

# # Khởi tạo YOLOv9pip
# # device = "cpu" # "cuda": GPU, "cpu": CPU, "mps:0"
# model  = DetectMultiBackend(weights="weights/yolov9-c-converted.pt", device=device, fuse=True )
# # model = YOLO('yolov8n-pose.pt')
# model  = AutoShape(model)

# # Load classname từ file classes.names
# with open("data_ext/classes.names") as f:
#     class_names = f.read().strip().split('\n')

# colors = np.random.randint(0,255, size=(len(class_names),3 ))
# tracks = []

# # Khởi tạo VideoCapture để đọc từ file video
# cap = cv2.VideoCapture(video_path)
# # cap = cv2.VideoCapture(0)

# # Tiến hành đọc từng frame từ video
# while True:
#     # Đọc
#     ret, frame = cap.read()
#     if not ret:
#         continue
#     # Đưa qua model để detect
#     results = model(frame)

#     detect = []
#     for detect_object in results.pred[0]:
#         label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
#         x1, y1, x2, y2 = map(int, bbox)
#         class_id = int(label)

#         if tracking_class is None:
#             if confidence < conf_threshold:
#                 continue
#         else:
#             if class_id != tracking_class or confidence < conf_threshold:
#                 continue

#         detect.append([ [x1, y1, x2-x1, y2 - y1], confidence, class_id ])


#     # Cập nhật,gán ID băằng DeepSort
#     tracks = tracker.update_tracks(detect, frame = frame)

#     # Vẽ lên màn hình các khung chữ nhật kèm ID
#     for track in tracks:
#         if track.is_confirmed():
#             track_id = track.track_id

#             # Lấy toạ độ, class_id để vẽ lên hình ảnh
#             ltrb = track.to_ltrb()
#             class_id = track.get_det_class()
#             x1, y1, x2, y2 = map(int, ltrb)
#             color = colors[class_id]
#             B, G, R = map(int,color)
#             confidence = float(confidence)
#             confidence = round(confidence, 2)
#             label = "{}-{}-{}".format(class_names[class_id], track_id, confidence)
#             # label = "{}-{}".format('Prs', track_id)

#             cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 1)
#             cv2.rectangle(frame, (x1 , y1 - 20), (x1 + len(label) * 8, y1), (B, G, R), -1)
#             cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255), 1)
        
#     cv2.imshow("DemoDtAIRC-H3", frame)
#     # Bấm Q thì thoát
#     if cv2.waitKey(1) == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()




 # Show hình ảnh lên màn hình
    # desired_width = 1600  # Độ rộng mong muốn của cửa sổ hiển thị
    # scale_factor = desired_width / frame.shape[1]  # Tính tỷ lệ scale dựa trên độ rộng mong muốn
    # desired_height = int(frame.shape[0] * scale_factor)  # Tính độ cao tương ứng

    # # Resize frame
    # resized_frame = cv2.resize(frame, (desired_width, desired_height))

    # # Hiển thị hình ảnh đã resize
    # cv2.imshow("DemoDtAIRC-H3", resized_frame)
    
    
    
# import cv2
# import torch
# import numpy as np
# from deep_sort_realtime.deepsort_tracker import DeepSort
# from models.common import DetectMultiBackend, AutoShape
# import mediapipe as mp

# # Initialize MediaPipe
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.is_available())
# print(torch.__version__)

# # Config values
# # video_path = 'data_ext/test6.mp4'
# video_path=0
# conf_threshold = 0.1
# tracking_class = 0 # None: track all

# # Initialize DeepSort
# tracker = DeepSort(max_age=20)

# # Initialize YOLOv9
# model  = DetectMultiBackend(weights="weights/yolov9-c.pt", device=device, fuse=True)
# model  = AutoShape(model)

# # Load class names from file
# with open("data_ext/classes.names") as f:
#     class_names = f.read().strip().split('\n')

# colors = np.random.randint(0,255, size=(len(class_names),3))
# tracks = []

# # Initialize VideoCapture to read from video file
# cap = cv2.VideoCapture(video_path)

# # Initialize MediaPipe Pose
# pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         continue
    
#     # Detect objects using YOLOv9
#     results = model(frame)

#     # Extract bounding boxes and class IDs
#     detect = []
#     for detect_object in results.pred[0]:
#         label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
#         x1, y1, x2, y2 = map(int, bbox)
#         class_id = int(label)

#         if tracking_class is None:
#             if confidence < conf_threshold:
#                 continue
#         else:
#             if class_id != tracking_class or confidence < conf_threshold:
#                 continue

#         detect.append([[x1, y1, x2-x1, y2 - y1], confidence, class_id])

#     # Update tracks using DeepSort
#     tracks = tracker.update_tracks(detect, frame=frame)

#     # Draw bounding boxes and IDs
#     for track in tracks:
#         if track.is_confirmed():
#             track_id = track.track_id

#             ltrb = track.to_ltrb()
#             class_id = track.get_det_class()
#             x1, y1, x2, y2 = map(int, ltrb)
#             color = colors[class_id]
#             B, G, R = map(int, color)

#             label = "{}-{}".format(class_names[class_id], track_id)

#             cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
#             cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
#             cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#     # Use MediaPipe to detect human poses
#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(image)
    
#     if results.pose_landmarks:
#         mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
#     # Show hình ảnh lên màn hình
#     # desired_width = 500  # Độ rộng mong muốn của cửa sổ hiển thị
#     # desired_height =900
#     # # scale_factor = desired_width / frame.shape[1]  # Tính tỷ lệ scale dựa trên độ rộng mong muốn
#     # # desired_height = int(frame.shape[0] * scale_factor)  # Tính độ cao tương ứng

#     # # Resize frame
#     # resized_frame = cv2.resize(frame, (desired_width, desired_height))

#     # # Hiển thị hình ảnh đã resize
#     # cv2.imshow("DemoDtAIRC-H3", resized_frame)

#     cv2.imshow("DemoDtAIRC-H3", frame)
#     if cv2.waitKey(1) == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(torch.__version__)

# Config values
video_path = "data_ext/people.mp4"
# video_path=0
conf_threshold = 0.2
tracking_class = 0  # None: track all
skip_frames = 4  # Process every 3rd frame
frame_count = 0

# Initialize DeepSort
tracker = DeepSort(max_age=25)

# Initialize YOLOv9
model = DetectMultiBackend(weights="weights/yolov9-c-converted.pt", device=device, fuse=True)
model = AutoShape(model)

# Load class names
with open("data_ext/classes.names") as f:
    class_names = f.read().strip().split('\n')

colors = np.random.randint(0, 255, size=(len(class_names), 3))
tracks = []

# Initialize VideoCapture to read from the video file
cap = cv2.VideoCapture(video_path)

# Set to keep track of appeared track IDs
appeared_ids = set()
people_count = 0

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
    tracks = tracker.update_tracks(detect, frame=frame)

    # Draw bounding boxes and IDs on the frame
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id

            # Only count if the track ID has not appeared before
            # if track_id not in appeared_ids:
            #     appeared_ids.add(track_id)
            #     if class_names[class_id] == "#":
            #         people_count += 1
                    # print("New person ID:", track_id)  # Just for debugging

            # Get coordinates and class_id to draw on the image
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            color = colors[class_id]
            B, G, R = map(int, color)
            confidence = float(confidence)
            confidence = round(confidence, 2)
            label = "{}-{}-{}".format(class_names[class_id], track_id, confidence)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (148, 238, 148), 1)
            cv2.rectangle(frame, (x1 , y1 - 20), (x1 + len(label) * 8, y1), (148, 238, 148), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255), 1)
        
    
    # cv2.putText(frame, f"People count: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("DemoDtAIRC-H3", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


# cv2.putText(frame, f"People count: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)