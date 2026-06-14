import cv2
import time
from insightface.app import FaceAnalysis
import onnxruntime as ort
import numpy as np

app = FaceAnalysis(
    name = "buffalo_l",
    providers = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
        ]
)

app.prepare(
    ctx_id = 0,
    det_size = (640, 640)
    )

cap = cv2.VideoCapture(0)
prev_time = time.time()

fps = 0


def detect_faces(frame):
    faces = app.get(frame)
    detection = []
    for face in faces:
        bbox = face.bbox.astype(int)
        landmarks = face.kps.astype(int)
        score = face.det_score
        detection.append({
            "bbox": bbox,
            "landmarks": landmarks,
            "score": score
        })
    return detection
if __name__ == "__main__":
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_faces(frame)

        for face in detections:
            bbox = face["bbox"].astype(int)
            landmarks = face["landmarks"]
            score = face["score"]

            x1, y1, x2, y2 = bbox

            color = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255)]
            for i, point in enumerate(landmarks):
                cv2.circle(frame, tuple(point.astype(int)), 2, color[i % len(color)], -1) 
                """1: Right Eye, 2: Left Eye, 3: Nose, 4: Right Mouth, 5: Left Mouth"""
                cv2.putText(
                    frame,
                    f"{i+1}",
                    tuple(point.astype(int) + [5, -5]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color[i % len(color)],
                    1
                )
                left_eye = landmarks[1]
                right_eye = landmarks[0]

                angle = 180 - np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
                print("Distance between eyes:", np.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]))
                print("Angle between eyes:", angle)

            cv2.line(
                frame,
                tuple(landmarks[0].astype(int)),
                tuple(landmarks[1].astype(int)),
                (255, 0, 0),
                1
            )

            center = (int((left_eye[0] + right_eye[0]) / 2), int((left_eye[1] + right_eye[1]) / 2))
            M = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
            aligned_face = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Faces: {len(face)}",
                (400, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(0, 255, 0), 2
            )

            cv2.putText(
                frame,
                f"Confidence: {face['score']:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,(0, 255, 0), 1
            )

        curr_time = time.time()
        instant_fps = 1 / (curr_time - prev_time)
        fps = 0.9 * fps + 0.1 * instant_fps
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Face Detection", frame)
        cv2.imshow("Aligned Face", aligned_face)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()