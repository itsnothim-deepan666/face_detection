import cv2
import numpy as np


def align_face(image, left_eye, right_eye, output_size=(160, 160)):
    if left_eye[0] > right_eye[0]:
        left_eye, right_eye = right_eye, left_eye

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    center = (
        int((left_eye[0] + right_eye[0]) / 2),
        int((left_eye[1] + right_eye[1]) / 2)
    )

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        M,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_CUBIC
    )

    eye_dist = np.linalg.norm(np.array(right_eye) - np.array(left_eye))

    crop_size = int(eye_dist * 2.5)

    x1 = max(0, center[0] - crop_size)
    y1 = max(0, center[1] - crop_size)
    x2 = min(rotated.shape[1], center[0] + crop_size)
    y2 = min(rotated.shape[0], center[1] + crop_size)

    aligned_face = rotated[y1:y2, x1:x2]

    if aligned_face.size == 0:
        return None

    aligned_face = cv2.resize(
        aligned_face,
        output_size,
        interpolation=cv2.INTER_LINEAR
    )

    return aligned_face