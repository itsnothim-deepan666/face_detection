import numpy as np
import os
import cv2

from align import align_face
from detect import detect_faces
from embed import embed

name = input("Enter the name of the person: ")
IMAGE_DIR = f"data/images/{name}"
SAVE_DIR = "data/embedding_database"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

embeddings = []

for filename in os.listdir(IMAGE_DIR):
    file_path = os.path.join(IMAGE_DIR, filename)

    image = cv2.imread(file_path)

    detections = detect_faces(image)

    if len(detections) == 0:
        print(f"No face detected in {filename}. Skipping.")
        continue

    det = detections[0]

    aligned_face = align_face(image, det["landmarks"][1], det["landmarks"][0])

    x1, y1, x2, y2 = det["bbox"].astype(int)
    face_crop = cv2.resize(aligned_face[y1:y2, x1:x2], (160, 160))

    embedding = embed(face_crop)
    embeddings.append(embedding.cpu().numpy())

embeddings = np.vstack(embeddings)

np.save(
    f"{SAVE_DIR}/{name}_embeddings.npy",
    embeddings
)

print(f"Saved {len(embeddings)} embeddings for {name} to {SAVE_DIR}/{name}_embeddings.npy")