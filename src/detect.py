import cv2
import time
from insightface.app import FaceAnalysis
import onnxruntime as ort

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        color = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255)]
        for i, point in enumerate(face.kps):
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

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (400, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,(0, 255, 0), 2
        )

        cv2.putText(
            frame,
            f"Confidence: {face.det_score:.2f}",
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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()