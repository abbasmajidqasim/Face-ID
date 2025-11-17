import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

yolo_model = YOLO("yolov11n-face.pt")
facenet = InceptionResnetV1(pretrained='vggface2').eval()

folder_path = "images"
image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

embeddings_list = []

for image_path in image_paths:
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ خطأ: لم يتم العثور على الصورة {image_path}!")
        continue

    results = yolo_model(image)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = image[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (160, 160))
            face_tensor = torch.tensor(face_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0

            with torch.no_grad():
                embedding = facenet(face_tensor).numpy()

            embeddings_list.append(embedding)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, "Face Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Detected Face", image)
    cv2.waitKey(500)

cv2.destroyAllWindows()

if embeddings_list:
    avg_embedding = np.mean(embeddings_list, axis=0)
    np.save("embeddings.npy", avg_embedding)
    print("✅ تم حفظ متوسط الـ embeddings في 'embeddings.npy'")
else:
    print("⚠️ لم يتم العثور على أي وجه!")
