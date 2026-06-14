from facenet_pytorch import InceptionResnetV1
import cv2
import numpy as np
import torch

def embed(image):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = InceptionResnetV1(
        pretrained='vggface2'
    ).eval().to(device)

    face_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_tensor = (torch.tensor(
        face_rgb,
        dtype=torch.float32
    ).permute(2, 0, 1)/255.0).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(face_tensor)

        print("Embedding shape:", embedding.shape)
        print("Embedding:", embedding)

        return embedding

