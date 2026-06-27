import os
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import glob
import time
import argparse
import numpy as np

from detect import detect_faces
from recognize import recognize_face

SAVE_EMBED_DIR = "data/embeddings"
SAVE_IMAGE_DIR = "data/images"

os.makedirs(SAVE_EMBED_DIR, exist_ok=True)

# Warmup InsightFace before camera
dummy = np.zeros((640, 640, 3), dtype=np.uint8)
detect_faces(dummy)

parser = argparse.ArgumentParser()

group = parser.add_mutually_exclusive_group(required=True)

group.add_argument("--register", type=str)
group.add_argument("--register_from_images", type=str)
group.add_argument("--recognize", type=str)

parser.add_argument("--camera-index", type=int, default=0)

args = parser.parse_args()


def open_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

    if cap.isOpened():
        print(f"Camera opened successfully at index {camera_index}")
        return cap

    print("Unable to open camera")
    return None


def process_image(image):
    detections = detect_faces(image)

    if len(detections) != 1:
        return None

    face = detections[0]

    bbox = face["bbox"].astype(int)
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
        return None

    embedding = face["embedding"]
    embedding = embedding / np.linalg.norm(embedding)

    if embedding.shape != (512,):
        return None

    return embedding, face_crop, face


def save_embeddings(name, embeddings):
    embeddings = np.vstack(embeddings)

    save_path = os.path.join(
        SAVE_EMBED_DIR,
        f"{name}_embeddings.npy"
    )

    np.save(save_path, embeddings)

    print(f"Saved {len(embeddings)} embeddings for {name}")


def register_from_webcam(name):
    cap = open_camera(args.camera_index)

    if cap is None:
        return

    embeddings = []
    required_samples = 10
    last_capture = 0

    print(f"Collecting {required_samples} samples for {name}")

    while len(embeddings) < required_samples:
        ret, frame = cap.read()

        if not ret:
            break

        result = process_image(frame)

        if result is not None:
            embedding, face_crop, _ = result

            if time.time() - last_capture > 0.5:
                embeddings.append(embedding)
                last_capture = time.time()

                cv2.imshow("Face Crop", face_crop)

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

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) == 0:
        print("No embeddings collected.")
        return

    save_embeddings(name, embeddings)


def register_from_images(name):
    person_dir = os.path.join(SAVE_IMAGE_DIR, name)
    image_paths = glob.glob(os.path.join(person_dir, "*"))

    if len(image_paths) == 0:
        print("No images found.")
        return

    embeddings = []

    for img_path in image_paths:
        image = cv2.imread(img_path)

        if image is None:
            continue

        result = process_image(image)

        if result is not None:
            embedding, _, _ = result
            embeddings.append(embedding)

    if len(embeddings) == 0:
        print("No valid embeddings.")
        return

    save_embeddings(name, embeddings)


def recognize(name):
    cap = open_camera(args.camera_index)

    if cap is None:
        return

    last_similarity = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        result = process_image(frame)

        if result is not None:
            embedding, face_crop, face = result

            match = recognize_face(name, embedding)

            if match is not None:
                matched_name, similarity = match
                last_similarity = similarity

            bbox = face["bbox"].astype(int)
            x1, y1, x2, y2 = bbox

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.imshow("Face Crop", face_crop)

        if last_similarity is not None:
            cv2.putText(
                frame,
                f"{name}: {last_similarity*100:.2f}%",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow("Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if args.register:
    register_from_webcam(args.register)

elif args.register_from_images:
    register_from_images(args.register_from_images)

elif args.recognize:
    recognize(args.recognize)