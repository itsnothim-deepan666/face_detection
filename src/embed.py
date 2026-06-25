from facenet_pytorch import InceptionResnetV1
import cv2
import torch
import numpy as np
import torch.nn.functional as F

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = InceptionResnetV1(
    pretrained='vggface2'
).eval().to(device)


def _prepare_face(image, target_size=(160, 160)):
    if image is None or image.size == 0:
        return None

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]
    if h < 10 or w < 10:
        return None

    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def prewhiten(x):
    x = x.astype(np.float32)
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    return (x - mean) / std_adj


def embed(image):
    prepared_face = _prepare_face(image)
    if prepared_face is None:
        return None
    face_rgb = cv2.cvtColor(prepared_face, cv2.COLOR_BGR2RGB)
    face = face_rgb.astype(np.float32)
    face = prewhiten(face)

    face_tensor = torch.from_numpy(face.astype(np.float32)) \
                   .permute(2, 0, 1) \
                   .unsqueeze(0) \
                   .to(device)

    with torch.no_grad():
        embedding = model(face_tensor)

        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding.cpu().numpy()[0]