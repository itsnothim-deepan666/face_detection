import cv2
import numpy as np

def align_face(image, left_eye, right_eye):
    # Calculate the angle between the eyes
    angle = 180 - np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    print("Distance between eyes:", np.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]))
    print("Angle between eyes:", angle)

    # Calculate the center point between the eyes
    center = (int((left_eye[0] + right_eye[0]) / 2), int((left_eye[1] + right_eye[1]) / 2))

    # Get the rotation matrix for the desired angle
    M = cv2.getRotationMatrix2D(tuple(center), -angle, 1.0)

    # Rotate the image to align the face
    aligned_face = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return aligned_face