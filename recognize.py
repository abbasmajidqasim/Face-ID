import cv2
import numpy as np
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from scipy.spatial.distance import cosine
import torch
import os


# تحميل قاعدة البيانات
DB_FILE = "vit_embeddings.npy"
if not os.path.exists(DB_FILE):
    print("❌ لا يوجد قاعدة بيانات! سجل مستخدم أولاً.")
    exit()

face_db = np.load(DB_FILE, allow_pickle=True).item()


# تحميل YOLO
yolo = YOLO("yolov11n-face.pt")

# تحميل ViT
processor = AutoImageProcessor.from_pretrained("google/vit-large-patch16-224-in21k")
vit = AutoModel.from_pretrained("google/vit-large-patch16-224-in21k").eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
vit.to(device)

# دالة استخراج تمثيلات ViT
def get_vit_embedding(face_img):
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)

    inputs = processor(images=pil_img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = vit(**inputs)

    return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()


def match_face(embedding, threshold=0.75):
    best_name = "Unknown"
    best_score = -1

    for name, saved_emb in face_db.items():
        sim = 1 - cosine(embedding, saved_emb)
        if sim > best_score:
            best_score = sim
            best_name = name

    return best_name, best_score


cap = cv2.VideoCapture(0)
print("🚀 نظام التعرف باستخدام ViT جاهز...")


while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = yolo(frame, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        emb = get_vit_embedding(face)
        name, score = match_face(emb)

        is_match = score > 0.50
        color = (0,255,0) if is_match else (0,0,255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("ViT Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
