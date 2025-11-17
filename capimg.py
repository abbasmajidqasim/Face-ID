import cv2
import os
import requests
import numpy as np

# عنوان الـ IP الخاص بكاميرا ESP32-CAM
esp32_cam_ip = "http://192.168.8.236/capture"  # استبدل بالـ IP الذي ذكرته

# دالة لتحميل الصورة من كاميرا ESP32-CAM
def get_image_from_esp32():
    try:
        # إرسال طلب GET للحصول على الصورة
        response = requests.get(esp32_cam_ip, stream=True)
        if response.status_code == 200:
            # تحويل الصورة إلى مصفوفة NumPy باستخدام OpenCV
            img_array = np.array(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, -1)  # -1 تعني حفظ الصورة بنفس صيغة الملف
            return img
        else:
            print(f"⚠️ تعذر الوصول إلى الكاميرا. رمز الاستجابة: {response.status_code}")
            return None
    except Exception as e:
        print(f"⚠️ حدث خطأ: {e}")
        return None

num_captures = 0
max_captures = 10
output_folder = "images"
os.makedirs(output_folder, exist_ok=True)

print("📸 اضغط على 'q' لالتقاط صورة. اضغط على 'ESC' للخروج.")

while num_captures < max_captures:
    frame = get_image_from_esp32()

    if frame is None:
        print("⚠️ فشل في الحصول على صورة من كاميرا ESP32-CAM.")
        break

    cv2.imshow("الكاميرا", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        filename = os.path.join(output_folder, f"capture_{num_captures + 1}.jpg")
        cv2.imwrite(filename, frame)
        print(f"✅ تم التقاط الصورة {num_captures + 1} وحفظها في {filename}")
        num_captures += 1

    elif key == 27:  # زر ESC
        print("🚪 تم الخروج من البرنامج.")
        break

cv2.destroyAllWindows()
