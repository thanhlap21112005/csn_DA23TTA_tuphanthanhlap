import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm

DATA_DIR = 'dataset'
CACHE_DIR = 'cache'
EMB_DIR = 'embeddings'

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EMB_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

embeddings = []
labels = []

persons = [p for p in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, p))]
if not persons:
    print("Thư mục dataset/ đang trống. Hãy tạo dataset/{ten_nguoi}/anh.jpg trước.")
    exit(0)

for person in sorted(persons):
    person_dir = os.path.join(DATA_DIR, person)
    image_files = [f for f in os.listdir(person_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"Khong tim thay anh trong {person_dir}, bo qua.")
        continue

    print(f"Xu ly nguoi: {person} ({len(image_files)} anh)")
    for img_name in tqdm(image_files, desc=f"{person}", unit="img"):
        img_path = os.path.join(person_dir, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print("Loi mo anh, bo qua", img_path, e)
            continue

        face = mtcnn(img)
        if face is None:
            print("Khong detect duoc mat trong", img_path)
            continue

        with torch.no_grad():
            face = face.unsqueeze(0).to(device)
            emb = resnet(face).cpu().numpy()

        embeddings.append(emb.flatten())
        labels.append(person)

if not embeddings:
    print("Khong co embedding nao duoc tao. Kiem tra lai dataset/")
    exit(0)

embeddings = np.vstack(embeddings)
labels = np.array(labels)

np.save(os.path.join(EMB_DIR, 'embeddings.npy'), embeddings)
np.save(os.path.join(EMB_DIR, 'labels.npy'), labels)

print("Da luu embeddings:", embeddings.shape)
print("So luong mau:", len(labels))
