# import cv2
# import torch
# import numpy as np
# from deep_sort_realtime.deepsort_tracker import DeepSort
# from models.common import DetectMultiBackend, AutoShape
# import os

# # Kiểm tra GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.is_available())
# print(torch.__version__)

# # Khởi tạo DeepSort
# tracker = DeepSort(max_age=15)

# # Cấu hình các giá trị
# video_path = "data_ext/people.mp4"
# conf_threshold = 0.1
# tracking_class = 0 # None: track all


# # Khởi tạo model detect
# model = DetectMultiBackend(weights="weights/yolov9-c-converted.pt", device=device, fuse=True)
# model = AutoShape(model)

# # Load classname từ file classes.names
# with open("data_ext/classes.names") as f:
#     class_names = f.read().strip().split('\n')

# # Tạo mảng màu ngẫu nhiên cho việc đánh dấu lớp
# colors = np.random.randint(0, 255, size=(len(class_names), 3))

# # Khởi tạo VideoCapture để đọc từ file video
# cap = cv2.VideoCapture(video_path)

# # Tạo thư mục để lưu ảnh kết quả
# output_dir = "output_images"
# os.makedirs(output_dir, exist_ok=True)

# # Biến đếm cho tên các ảnh kết quả
# count = 0

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

#         detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

#     # Cập nhật, gán ID bằng DeepSort
#     tracks = tracker.update_tracks(detect, frame=frame)

#     # Vẽ lên màn hình các khung chữ nhật kèm ID và cắt phần người detect
#     for track in tracks:
#         if track.is_confirmed():
#             track_id = track.track_id

#             # Lấy tọa độ, class_id để vẽ lên hình ảnh và cắt phần người detect
#             ltrb = track.to_tlbr() # Lấy tọa độ tuyệt đối của bounding box
#             class_id = track.get_det_class()
#             x1, y1, x2, y2 = map(int, ltrb)
#             color = colors[class_id]
#             B, G, R = map(int, color)

#             label = "{}-{}".format(class_names[class_id], track_id)

#             cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 1)
#             cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 9, y1), (B, G, R), -1)
#             cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255), 1)
            
#             # Cắt phần người detect từ frame
#             person_img = frame[y1:y2, x1:x2]
            
#             # Lưu frame chứa bounding box và ID vào tệp ảnh
#             output_path = os.path.join(output_dir, f"{track_id}_{count}.jpg")
#             cv2.imwrite(output_path, person_img)
#             count += 1
    
#     cv2.imshow("DemoDtAIRC-H3", frame)

#     # Bấm Q thì thoát
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
video_path = "data_ext/videoclass.mp4"
# video_path=0
conf_threshold = 0.2
tracking_class = 0  # None: track all
skip_frames = 3  # Process every 3rd frame
resize_width = 1280  # Adjust based on your needs
resize_height = 720  # Adjust based on your needs
frame_count = 0

# Initialize DeepSort
tracker = DeepSort(max_age=5)

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
    
    if frame_count == 0:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        resize_height = int((resize_width / frame_width) * frame_height)

    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    # Resize frame
    frame = cv2.resize(frame, (resize_width, resize_height))

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

    # Count number of "person" detections
    num_persons = sum(1 for obj in detect if obj[2] == 0)  # Assuming class_id of "person" is 0

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
            label = "ID-{} : {}".format(track_id, class_names[class_id]) #confidence

            cv2.rectangle(frame, (x1, y1), (x2, y2), (148, 238, 148), 1)
            cv2.rectangle(frame, (x1 , y1 - 20), (x1 + len(label) * 11, y1), (148, 238, 148), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        
    # Print number of "person" detections in the frame
    num_persons_text = f"Number prs: {num_persons}"
    cv2.putText(frame, num_persons_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("DemoDtAIRC-H3", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

