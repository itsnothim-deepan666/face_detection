import os
import cv2
import glob
import argparse
import numpy as np

from detect import detect_faces
from align import align_face
from embed import embed
from recognize import recognize_face


SAVE_EMBED_DIR = "data/embeddings"
SAVE_IMAGE_DIR = "data/images"

os.makedirs(SAVE_EMBED_DIR, exist_ok=True)


# -----------------------------
# ARGUMENTS
# -----------------------------
parser = argparse.ArgumentParser()

group = parser.add_mutually_exclusive_group(required=True)

group.add_argument(
    "--register",
    type=str,
    help="Register a person from webcam"
)

group.add_argument(
    "--register_from_images",
    type=str,
    help="Register a person from data/images/{name}"
)

group.add_argument(
    "--recognize",
    type=str,
    help="Recognize a person from webcam"
)

args = parser.parse_args()


# -----------------------------
# COMMON PROCESSOR
# -----------------------------
def process_image(image):
    detections = detect_faces(image)

    if len(detections) != 1:
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

    embedding = embed(aligned_face)

    return embedding, aligned_face, face


# -----------------------------
# REGISTER FROM WEBCAM
# -----------------------------
def register_from_webcam(name):
    cap = cv2.VideoCapture(0)

    embeddings = []
    required_samples = 10

    print(f"Collecting {required_samples} samples for {name}")

    while len(embeddings) < required_samples:
        ret, frame = cap.read()
        if not ret:
            break

        result = process_image(frame)

        if result is not None:
            embedding, aligned_face, _ = result
            embeddings.append(embedding)

            cv2.imshow("Aligned Face", aligned_face)

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
        print("No embeddings collected.")
        return

    embeddings = np.vstack(embeddings)

    save_path = os.path.join(
        SAVE_EMBED_DIR,
        f"{name}_embeddings.npy"
    )

    np.save(save_path, embeddings)

    print(f"Saved {len(embeddings)} embeddings for {name}")


# -----------------------------
# REGISTER FROM IMAGES
# -----------------------------
def register_from_images(name):
    person_dir = os.path.join(SAVE_IMAGE_DIR, name)

    image_paths = glob.glob(
        os.path.join(person_dir, "*")
    )

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

    embeddings = np.vstack(embeddings)

    save_path = os.path.join(
        SAVE_EMBED_DIR,
        f"{name}_embeddings.npy"
    )

    np.save(save_path, embeddings)

    print(f"Saved {len(embeddings)} embeddings for {name}")


# -----------------------------
# RECOGNIZE
# -----------------------------
def recognize(name):
    cap = cv2.VideoCapture(0)

    frame_count = 0
    last_similarity = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % 10 == 0:
            result = process_image(frame)

            if result is not None:
                embedding, aligned_face, face = result

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

                cv2.imshow("Aligned Face", aligned_face)

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

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# -----------------------------
# MODE ROUTING
# -----------------------------
if args.register:
    register_from_webcam(args.register)

elif args.register_from_images:
    register_from_images(args.register_from_images)

elif args.recognize:
    recognize(args.recognize)