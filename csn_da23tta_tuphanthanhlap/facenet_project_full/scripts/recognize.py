import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import joblib
import numpy as np
from PIL import Image
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

model_path = 'models/classifier.joblib'
if not os.path.exists(model_path):
    print("Khong tim thay models/classifier.joblib. Hay chay train_classifier.py truoc.")
    exit(0)

clf_data = joblib.load(model_path)
clf = clf_data['model']
le = clf_data['label_encoder']

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Khong mo duoc webcam. Kiem tra lai camera.")
    exit(0)

THRESHOLD_UNKNOWN = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        print("Khong doc duoc frame tu webcam.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    boxes, probs = mtcnn.detect(pil_img)
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = max(0, x2), max(0, y2)

            face = pil_img.crop((x1, y1, x2, y2)).resize((160, 160))
            face_np = np.array(face).astype(np.float32) / 255.0
            face_np = (face_np - 0.5) / 0.5
            face_tensor = torch.from_numpy(face_np).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = resnet(face_tensor).cpu().numpy()

            probs = clf.predict_proba(emb)[0]
            idx = np.argmax(probs)
            conf = probs[idx]
            name = le.inverse_transform([idx])[0]

            if conf < THRESHOLD_UNKNOWN:
                disp_name = f"unknown ({conf:.2f})"
            else:
                disp_name = f"{name} ({conf:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, disp_name, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('FaceNet Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
