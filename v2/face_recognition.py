"""
Setup:
    pip install opencv-contrib-python
    Download models into the same directory:
        wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
        wget https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx

Usage:
    # Register a new face:
    python face_recognition.py --register "Alice"

    # Run live recognition:
    python face_recognition.py
"""

import cv2
import numpy as np
import pickle
import argparse
import os
import sys
import time

DETECTION_MODEL  = "face_detection_yunet_2023mar.onnx"
RECOGNITION_MODEL = "face_recognition_sface_2021dec.onnx"
DB_FILE          = "faces.pkl"          # persisted face database
COSINE_THRESHOLD = 0.363               # SFace recommended threshold (lower = stricter)
FRAME_SCALE      = 0.5                 # downscale input for detection (speeds up Pi)
SKIP_FRAMES      = 2                   # run detection every N frames, track in between
DISPLAY_W        = 640
DISPLAY_H        = 480


def build_detector(w, h):
    det = cv2.FaceDetectorYN.create(
        DETECTION_MODEL, "", (w, h),
        score_threshold=0.7,    # raise to cut false positives
        nms_threshold=0.3,
        top_k=5,                # Pi: cap to 5 faces max
    )
    return det


def build_recognizer():
    return cv2.FaceRecognizerSF.create(RECOGNITION_MODEL, "")


def load_db():
    """{ name: [embedding, ...] }"""
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_db(db):
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)


def embed_face(recognizer, frame, face_box):
    """Align + extract 128-d feature vector for one face."""
    aligned = recognizer.alignCrop(frame, face_box)
    feat = recognizer.feature(aligned)
    return feat  # shape (1, 128)


def best_match(recognizer, query_feat, db):
    """Return (name, score). score < COSINE_THRESHOLD → unknown."""
    best_name, best_score = "Unknown", 0.0
    for name, embeddings in db.items():
        for ref_feat in embeddings:
            score = recognizer.match(query_feat, ref_feat,
                                     cv2.FaceRecognizerSF_FR_COSINE)
            if score > best_score:
                best_score, best_name = score, name
    if best_score < COSINE_THRESHOLD:
        return "Unknown", best_score
    return best_name, best_score


def draw_face(frame, box, label, score, color=(0, 220, 0)):
    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    text = f"{label} {score:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y), color, -1)
    cv2.putText(frame, text, (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)



def register_face(name):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_H)

    det_w = int(DISPLAY_W * FRAME_SCALE)
    det_h = int(DISPLAY_H * FRAME_SCALE)
    detector   = build_detector(det_w, det_h)
    recognizer = build_recognizer()
    db = load_db()

    collected, needed = [], 5
    print(f"[register] Look at the camera. Collecting {needed} samples for '{name}'...")

    while len(collected) < needed:
        ok, frame = cap.read()
        if not ok:
            break

        small = cv2.resize(frame, (det_w, det_h))
        _, faces = detector.detect(small)

        if faces is None or len(faces) == 0:
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # pick the most confident face
            face = sorted(faces, key=lambda f: f[14], reverse=True)[0]

            # Scale bbox (0:4) AND all 5 landmarks (4:14) to full-res.
            # Leave confidence (index 14) untouched.
            # alignCrop() uses landmarks for affine alignment — wrong coords
            # produce garbage embeddings.
            scaled = face.copy()
            scaled[:14] /= FRAME_SCALE

            feat = embed_face(recognizer, frame, scaled)
            collected.append(feat)

            draw_face(frame, scaled, f"Sample {len(collected)}/{needed}", face[14])

        cv2.imshow("Register", frame)
        if cv2.waitKey(200) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if collected:
        db.setdefault(name, []).extend(collected)
        save_db(db)
        print(f"[register] Saved {len(collected)} embeddings for '{name}'. DB now has {len(db)} identities.")
    else:
        print("[register] Nothing collected.")



def run_recognition():
    db = load_db()
    if not db:
        sys.exit("[error] Face database is empty. Register faces first with --register <name>")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    det_w = int(DISPLAY_W * FRAME_SCALE)
    det_h = int(DISPLAY_H * FRAME_SCALE)
    detector   = build_detector(det_w, det_h)
    recognizer = build_recognizer()

    frame_idx   = 0
    last_results = []   # (box_fullres, label, score)
    fps_t = time.time()
    fps   = 0.0

    print("[live] Running. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % SKIP_FRAMES == 0:
            small = cv2.resize(frame, (det_w, det_h))
            _, faces = detector.detect(small)
            last_results = []

            if faces is not None:
                for face in faces:
                    # Scale bbox (0:4) AND all 5 landmarks (4:14) to full-res.
                    # Leave confidence (index 14) untouched.
                    box = face.copy()
                    box[:14] /= FRAME_SCALE

                    feat = embed_face(recognizer, frame, box)
                    name, score = best_match(recognizer, feat, db)
                    last_results.append((box, name, score))

        for box, name, score in last_results:
            color = (0, 220, 0) if name != "Unknown" else (0, 60, 220)
            draw_face(frame, box, name, score, color)

        frame_idx += 1
        if frame_idx % 15 == 0:
            fps = 15 / (time.time() - fps_t)
            fps_t = time.time()
        cv2.putText(frame, f"FPS {fps:.1f}", (DISPLAY_W - 90, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YuNet + SFace face recognition (pure OpenCV)")
    parser.add_argument("--register", metavar="NAME",
                        help="Register a new face under this name")
    parser.add_argument("--list", action="store_true",
                        help="List registered identities")
    parser.add_argument("--delete", metavar="NAME",
                        help="Remove an identity from the database")
    args = parser.parse_args()

    # quick model check
    for path in (DETECTION_MODEL, RECOGNITION_MODEL):
        if not os.path.exists(path):
            sys.exit(f"[error] Model not found: {path}\n"
                     f"        Download from opencv_zoo — see script header.")

    if args.register:
        register_face(args.register)
    elif args.list:
        db = load_db()
        if not db:
            print("Database is empty.")
        else:
            for name, embs in db.items():
                print(f"  {name}: {len(embs)} sample(s)")
    elif args.delete:
        db = load_db()
        if args.delete in db:
            del db[args.delete]
            save_db(db)
            print(f"Deleted '{args.delete}'.")
        else:
            print(f"'{args.delete}' not found in database.")
    else:
        run_recognition()