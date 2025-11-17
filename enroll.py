import cv2
import numpy as np
import os
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch


# =============================
# تحميل YOLO
# =============================
yolo = YOLO("yolov11n-face.pt")

# =============================
# تحميل ViT
# =============================
processor = AutoImageProcessor.from_pretrained("google/vit-large-patch16-224-in21k")
vit = AutoModel.from_pretrained("google/vit-large-patch16-224-in21k").eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
vit.to(device)

DB_FILE = "vit_embeddings.npy"

# تحميل قاعدة البيانات
if os.path.exists(DB_FILE):
    face_db = np.load(DB_FILE, allow_pickle=True).item()
else:
    face_db = {}

# أخذ اسم الشخص
name = input("📝 ادخل اسم الشخص: ").strip()

cap = cv2.VideoCapture(0)
embeddings = []

print("📸 وجّه وجهك للكاميرا — اضغط S لالتقاط embedding و ESC للخروج")

# =============================
# دالة استخراج تمثيل ViT
# =============================
def get_vit_embedding(face_img):
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)

    inputs = processor(images=pil_img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = vit(**inputs)

    emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    return emb


# =============================
# حلقة التقاط التسجيل
# =============================
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

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imshow("Enrollment - ViT", frame)

    key = cv2.waitKey(1)

    if key == ord('s') and face.size != 0:
        emb = get_vit_embedding(face)
        embeddings.append(emb)
        print("✔ تم التقاط embedding")

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

if embeddings:
    face_db[name] = np.mean(embeddings, axis=0)
    np.save(DB_FILE, face_db)
    print(f"🔥 تم حفظ {name} داخل قاعدة البيانات!")
else:
    print("❌ لم يتم التقاط أي شيء")
