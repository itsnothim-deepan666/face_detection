# Real-Time Face Recognition System

This project is a real-time face recognition system built using **SCRFD (InsightFace)**, **OpenCV**, and **FaceNet**.

The system takes a live webcam feed, detects faces, extracts facial landmarks, aligns the face properly, generates embeddings, stores known identities, and compares live embeddings for recognition.

The purpose of this project is not just to build a working face recognition pipeline, but to understand how modern face recognition systems work internally.

---

# Project Goal

The goal of this project is to build a complete face recognition pipeline that can:

* Detect faces in real time
* Extract important facial landmarks
* Align faces for consistency
* Crop and preprocess face images
* Convert faces into embeddings
* Register known identities into a database
* Compare embeddings for recognition
* Reject unknown faces
* Log recognized individuals for attendance or access control

This project is being built step-by-step with focus on both implementation and understanding.

---

# How the System Works

Current pipeline:

```text id="1u1zzy"
Webcam Feed
   ↓
Face Detection (SCRFD)
   ↓
Bounding Box + Landmarks
   ↓
Face Alignment
   ↓
Face Crop
   ↓
Preprocessing
   ↓
FaceNet Embedding
   ↓
Database Matching
   ↓
Identity Recognition
   ↓
Attendance Logging (Upcoming)
```

At the current stage, the system can detect faces, align them, generate embeddings, register identities, and recognize known faces.

---

# Tools and Libraries Used

* **Python 3.10**
* **OpenCV**
* **InsightFace (SCRFD)**
* **NumPy**
* **PyTorch**
* **FaceNet (facenet-pytorch)**
* **ONNX Runtime**
* **Scikit-learn**

---

# What Has Been Built So Far

## Environment Setup

A dedicated Python virtual environment was created to isolate dependencies.

Installed:

```bash id="zkgnfx"
pip install insightface
pip install onnxruntime-gpu
pip install opencv-python==4.9.0.80
pip install numpy==1.26.4
pip install facenet-pytorch
pip install scikit-learn
```

GPU acceleration was attempted, but due to unresolved CUDA runtime dependency issues, the system currently runs on CPU.

Current performance:

```text id="5r1y40"
~3 FPS
```

---

## Webcam Input

The webcam feed is captured using:

```python id="jlwmx6"
cap = cv2.VideoCapture(0)
```

This acts as the live input source.

---

## Face Detection

SCRFD detects faces and returns:

* Bounding boxes
* Facial landmarks
* Detection confidence

Example:

```text id="n2e5od"
Bounding Box → [x1, y1, x2, y2]
Confidence → 0.98
```

---

## Multi-Face Detection

Multiple faces can be processed simultaneously:

```python id="eqflql"
for face in faces:
```

This makes the system scalable.

---

## FPS Tracking

Raw FPS:

```python id="ftwz6n"
fps = 1 / (curr_time - prev_time)
```

Smoothed FPS:

```python id="m9m3x2"
fps = 0.9 * fps + 0.1 * instant_fps
```

This stabilizes the displayed performance.

---

## Facial Landmark Detection

SCRFD provides:

* Left eye
* Right eye
* Nose
* Left mouth corner
* Right mouth corner

These are used to estimate face orientation.

---

## Face Alignment

The angle between the eyes is calculated:

```python id="0m0qxh"
angle = np.degrees(np.arctan2(dy, dx))
```

The frame is rotated:

```python id="84ig1g"
cv2.getRotationMatrix2D(center, -angle, 1.0)
```

This aligns the face horizontally and improves recognition consistency.

---

## Face Cropping

After alignment:

```python id="9xg3pa"
face_crop = aligned[y1:y2, x1:x2]
```

This isolates only the face.

---

## Face Resizing

Each cropped face is resized:

```python id="qmx8rd"
cv2.resize(face_crop, (160,160))
```

This matches FaceNet’s input size.

---

## Face Embedding Generation

FaceNet converts the processed face into a **512-dimensional embedding**.

Model:

```python id="8s2kec"
model = InceptionResnetV1(
    pretrained='vggface2'
).eval()
```

Output:

```text id="zcywvh"
torch.Size([1,512])
```

This embedding represents the identity of the person.

---

## Identity Registration

Known people can now be registered.

Instead of storing one image, multiple samples are stored.

Example:

```text id="v4mvlx"
data/
└── john/
    ├── img1.jpg
    ├── img2.jpg
    ├── img3.jpg
```

Each image is processed through:

```text id="xjlwmh"
Detect
↓
Align
↓
Crop
↓
Embed
```

All embeddings are stored together:

```text id="2g8g7d"
database/john.npy
```

Shape:

```text id="5g8ptn"
(3,512)
```

This improves recognition robustness.

---

## Face Recognition

Live embeddings are compared with stored embeddings.

Similarity is measured using cosine similarity.

Process:

```text id="k5sztf"
Live Face
↓
Embedding
↓
Compare with Database
↓
Best Match
```

Example:

```text id="jjlwm8"
Stored:
0.89
0.84
0.91
```

Best match:

```text id="vwvdfg"
0.91
```

If similarity is above threshold:

```text id="4ye5yr"
Recognized
```

Else:

```text id="w1gxy7"
Unknown
```

Current threshold:

```text id="fd56e5"
0.75
```

---

# Important Concepts Learned

## Face Detection

Locating faces in an image.

---

## Facial Landmarks

Finding important facial keypoints.

---

## Face Alignment

Reducing pose variation using eye geometry.

---

## Rotation Matrices

Used to rotate tilted faces.

---

## Tensors

Converting images into neural-network-compatible numerical data.

---

## Embeddings

Representing facial identity as a fixed-size vector.

---

## Gradient-Free Inference

Using:

```python id="5q1yhj"
with torch.no_grad():
```

to reduce memory usage and speed up inference.

---

## Cosine Similarity

Comparing embeddings by measuring directional similarity.

This is the core of recognition.

---

# What Comes Next

The next steps are:

* Multi-person recognition
* Multiple identity support
* Unknown face rejection tuning
* Attendance logging
* Evaluation and benchmarking
* GPU optimization

---

# Research Papers Referenced

* SCRFD — *Sample and Computation Redistribution for Efficient Face Detection*
* RetinaFace — *Single-stage Dense Face Localisation in the Wild*
* FaceNet — *A Unified Embedding for Face Recognition and Clustering*

---

# Final Goal

The final system should be able to:

* Recognize multiple people in real time
* Reject unknown faces
* Maintain attendance logs
* Work in smart access systems
* Integrate into robotics perception pipelines

The long-term aim is to understand and build a production-level face recognition pipeline from scratch.
