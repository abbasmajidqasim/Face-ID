import cv2
import torch
import numpy as np
import requests
from scipy.spatial.distance import cosine
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

# تحميل النماذج
yolo_model = YOLO("yolov11n-face.pt")
facenet = InceptionResnetV1(pretrained='vggface2').eval().to("cuda" if torch.cuda.is_available() else "cpu")
device = next(facenet.parameters()).device

# تحميل التضمينات المحفوظة
try:
    saved_embeddings = np.load("embeddings.npy")
except FileNotFoundError:
    print("❌ ملف embeddings.npy غير موجود!")
    exit()

def get_embedding(face):
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (160, 160))
    tensor = torch.tensor(face_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = tensor.to(device)
    with torch.no_grad():
        return facenet(tensor).cpu().numpy().flatten()

# عنوان الصورة من ESP32-CAM
url = "http://192.168.8.236/capture"

print("🚀 بدء التعرف السريع على الوجوه...")
while True:
    try:
        r = requests.get(url, timeout=2)
        if r.status_code != 200:
            print(f"⚠️ فشل التحميل: {r.status_code}")
            continue

        img_array = np.frombuffer(r.content, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            print("⚠️ صورة غير صالحة")
            continue

        results = yolo_model(frame, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            embedding = get_embedding(face)
            similarities = [1 - cosine(embedding, e) for e in saved_embeddings]
            best = max(similarities)

            match = 1 if best > 0.75 else 0
            color = (0, 255, 0) if match else (0, 0, 255)
            label = f"{best:.2f}"

            try:
                requests.get(f"http://192.168.8.236/result?match={match}", timeout=1)
                print(f"📡 Match sent: {match} ({label})")
            except Exception as e:
                print(f"⚠️ فشل الإرسال: {e}")

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Esc key
            break

    except Exception as e:
        print(f"❌ خطأ: {e}")
        continue

cv2.destroyAllWindows()
