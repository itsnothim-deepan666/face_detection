import os

cuda_dirs = [
    r"C:\Users\Deepankuran M\Builds\face_detect\face_detect\lib\site-packages\nvidia\cublas\bin",
    r"C:\Users\Deepankuran M\Builds\face_detect\face_detect\lib\site-packages\nvidia\cuda_runtime\bin",
    r"C:\Users\Deepankuran M\Builds\face_detect\face_detect\lib\site-packages\nvidia\cuda_nvrtc\bin",
    r"C:\Users\Deepankuran M\Builds\face_detect\face_detect\lib\site-packages\nvidia\cudnn\bin",
]

for d in cuda_dirs:
    if os.path.isdir(d):
        os.add_dll_directory(d)

import onnxruntime as ort

print("Available:", ort.get_available_providers())

model = r"C:\Users\Deepankuran M\.insightface\models\buffalo_l\det_10g.onnx"

sess = ort.InferenceSession(
    model,
    providers=["CUDAExecutionProvider"]
)

print("Loaded with:", sess.get_providers())