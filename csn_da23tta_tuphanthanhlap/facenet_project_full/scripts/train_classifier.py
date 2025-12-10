import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

EMB_DIR = 'embeddings'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

emb_path = os.path.join(EMB_DIR, 'embeddings.npy')
lbl_path = os.path.join(EMB_DIR, 'labels.npy')

if not (os.path.exists(emb_path) and os.path.exists(lbl_path)):
    print("Chua co file embeddings.npy hoac labels.npy. Hay chay extract_embeddings.py truoc.")
    exit(0)

X = np.load(emb_path)
y = np.load(lbl_path)

print("Shape X:", X.shape)
print("So luong nhan:", len(y))

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
target_names = le.inverse_transform(sorted(set(y_enc)))
print("=== Classification report ===")
print(classification_report(y_test, pred, target_names=target_names))

joblib.dump({'model': clf, 'label_encoder': le}, os.path.join(MODEL_DIR, 'classifier.joblib'))
print("Da luu model tai models/classifier.joblib")
