from facenet_pytorch import InceptionResnetV1
import cv2
import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = InceptionResnetV1(
    pretrained='vggface2'
).eval().to(device)


def _prepare_face(image, target_size=(160, 160)):
    if image is None or image.size == 0:
        return None

    # Ensure a 3-channel face image for the embedding model.
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]
    if h < 10 or w < 10:
        return None

    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

def embed(image):
    prepared_face = _prepare_face(image)
    if prepared_face is None:
        return None

    face_rgb = cv2.cvtColor(prepared_face, cv2.COLOR_BGR2RGB)

    face_tensor = (torch.tensor(
        face_rgb,
        dtype=torch.float32
    ).permute(2, 0, 1)/255.0).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(face_tensor)

        print("Embedding shape:", embedding.shape)
        print("Embedding:", embedding)

        return embedding

