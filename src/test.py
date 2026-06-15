from detect import detect_faces
from align import align_face
from embed import embed
from recognize import recognize_face
import cv2
import numpy as np

name = input("Enter the name of the person to recognize: ")
cap = cv2.VideoCapture(0)
a = True

while a == True:
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

            aligned_face = align_face(frame, left_eye, right_eye)

        cv2.line(
            frame,
            tuple(landmarks[0].astype(int)),
            tuple(landmarks[1].astype(int)),
            (255, 0, 0),
            1
        )

        cv2.putText(
            frame,
            f"Faces: {len(detections)}",
            (400, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,(0, 255, 0), 2
        )

        

        left_eye = landmarks[1]
        right_eye = landmarks[0]

        cv2.putText(
            frame,
            f"Confidence: {face['score']:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,(0, 255, 0), 1
        )
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.line
        (
            frame,
            tuple(landmarks[0].astype(int)),
            tuple(landmarks[1].astype(int)),
            (255, 0, 0),
            1
        )

        embedding = embed(aligned_face[y1:y2, x1:x2])
        max_similarity = recognize_face(name, embedding)
        if max_similarity is not None:
            cv2.putText(
                frame,
                f"Similarity: {max_similarity*100:.2f}%",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
        else:
            print(f"Could not recognize {name}. Please register the person first.")
            a = False
            break

    cv2.imshow("Aligned Face", aligned_face[y1:y2, x1:x2])
    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()