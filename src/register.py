import os
import cv2
import glob
import argparse
import numpy as np

from detect import detect_faces
from align import align_face
from embed import embed

SAVE_EMBED_DIR = "data/embeddings"
SAVE_IMAGE_DIR = "data/images"

os.makedirs(SAVE_EMBED_DIR, exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--name",
    type=str,
    required=True,
    help="Name of person to register"
)

parser.add_argument(
    "--from-images",
    action="store_true",
    help="Use existing images from data/images/{NAME}/"
)

args = parser.parse_args()

name = args.name.strip()
from_images = args.from_images

person_embed_path = os.path.join(
    SAVE_EMBED_DIR,
    f"{name}_embeddings.npy"
)

person_image_dir = os.path.join(
    SAVE_IMAGE_DIR,
    name
)


def process_image(image):
    detections = detect_faces(image)

    if len(detections) != 1:
        print("Skipping: requires exactly one face")
        return None

    face = detections[0]

    bbox = face["bbox"].astype(int)
    landmarks = face["landmarks"]

    x1, y1, x2, y2 = bbox

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    face_crop = image[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    if blur_score < 100:
        print("Skipping: blurry image")
        return None

    crop_landmarks = landmarks.copy()
    crop_landmarks[:, 0] -= x1
    crop_landmarks[:, 1] -= y1

    aligned_face = align_face(
        face_crop,
        crop_landmarks[1],
        crop_landmarks[0]
    )

    if aligned_face is None:
        return None

    return embed(aligned_face)


embeddings = []

if from_images:
    image_paths = glob.glob(
        os.path.join(person_image_dir, "*")
    )

    if len(image_paths) == 0:
        print(f"No images found in {person_image_dir}")
        exit()

    print(f"Processing {len(image_paths)} images...")

    for img_path in image_paths:
        image = cv2.imread(img_path)

        if image is None:
            print(f"Skipping unreadable file: {img_path}")
            continue

        embedding = process_image(image)

        if embedding is not None:
            embeddings.append(embedding)
            print(f"Processed: {img_path}")


else:
    cap = cv2.VideoCapture(0)

    required_samples = 10

    print(f"Collecting {required_samples} samples for {name}...")

    while len(embeddings) < required_samples:
        ret, frame = cap.read()

        if not ret:
            break

        embedding = process_image(frame)

        if embedding is not None:
            embeddings.append(embedding)

        cv2.putText(
            frame,
            f"Samples: {len(embeddings)}/{required_samples}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Register", frame)

        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if len(embeddings) == 0:
    print("No valid embeddings generated.")
    exit()

embeddings = np.vstack(embeddings)

np.save(person_embed_path, embeddings)

print(f"Saved {len(embeddings)} embeddings for {name}")
print(f"Saved at: {person_embed_path}")