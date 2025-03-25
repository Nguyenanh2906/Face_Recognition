import os
import cv2
import pickle
from insightface.app import FaceAnalysis

dataset_path = r"D:\face_recognition\dataset"

# Khởi tạo FaceAnalysis với cả detection và recognition
face_app = FaceAnalysis(
    providers=['CPUExecutionProvider', 'CUDAExecutionProvider'],
    allowed_modules=['detection', 'recognition']
)
face_app.prepare(ctx_id=0, det_size=(640, 640))

face_data_list = []

# Duyệt qua các thư mục con
for subdir in os.listdir(dataset_path):
    subdir_path = os.path.join(dataset_path, subdir)
    if not os.path.isdir(subdir_path):
        continue

    # Duyệt qua các ảnh
    for img_name in os.listdir(subdir_path):
        img_path = os.path.join(subdir_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Lỗi đọc ảnh: {img_path}")
            continue
        faces = face_app.get(img)
        if len(faces) == 0:
            print(f"Không tìm thấy khuôn mặt trong ảnh: {img_path}")
            continue

        face = faces[0]
        face_info = {
            "student_id": subdir,  # Sử dụng tên thư mục làm student_id
            "img_path": img_path,
            "bbox": [{"x": int(face.bbox[0]), "y": int(face.bbox[1])}, {"x": int(face.bbox[2]), "y": int(face.bbox[3])}],
            "landmarks": [{"x": int(kp[0]), "y": int(kp[1])} for kp in face.kps],
            # Lưu embedding
            "embeddings": face.embedding.tolist()  # Chuyển embedding sang list
        }
        face_data_list.append(face_info)

# Lưu ra file
with open("face_data_list.pkl", "wb") as f:
    pickle.dump(face_data_list, f)

print("✅ Hoàn tất, file 'face_data_list.pkl' đã có bounding box, landmarks, embedding.")