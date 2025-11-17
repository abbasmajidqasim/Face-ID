# 🔥 Face Recognition System using YOLOv11n-Face + ViT-Large

This project implements a high-accuracy Face Recognition system using:

- **YOLOv11n-Face** for face detection  
- **ViT-Large (google/vit-large-patch16-224-in21k)** for visual embeddings  
- **Cosine similarity** for comparing embeddings  
- **OpenCV** for webcam streaming  

This is one of the strongest open-source face recognition pipelines without requiring face-specific training.

---

## 🚀 Features

✔ Real-time face detection  
✔ Real-time face recognition  
✔ Can register multiple users  
✔ Embeddings stored in `.npy` database  
✔ Works on CPU/GPU  
✔ Very high accuracy using ViT embeddings  
✔ Simple & clean architecture  
✔ No C++/CUDA building required  

---

## 🛠️ Technologies Used

| Component | Technology |
|----------|------------|
| Face Detection | YOLOv11n-Face |
| Face Embedding | ViT-Large (HuggingFace) |
| Programming Language | Python |
| Webcam Access | OpenCV |
| Database | Numpy `.npy` file |
| Similarity Metric | Cosine Similarity |

---

## 📦 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/abbasmajidqasim/Face-ID.git
cd Face-ID