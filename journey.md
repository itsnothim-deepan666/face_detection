# Real-Time Face Recognition System

This project is a real-time face recognition system built using **SCRFD (InsightFace)**, **OpenCV**, and **FaceNet**.

The system captures a live webcam feed, detects faces, extracts facial landmarks, aligns faces, preprocesses them, generates embeddings, stores known identities, and compares live embeddings for recognition.

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

```text
Webcam Feed / Stored Images
          ↓
Face Detection (SCRFD)
          ↓
Bounding Box + Landmarks
          ↓
Face Alignment
          ↓
Blur Filtering
          ↓
Face Crop
          ↓
Prewhitening
          ↓
FaceNet Embedding
          ↓
L2 Normalization
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

* Python 3.10
* OpenCV
* InsightFace (SCRFD)
* NumPy
* PyTorch
* FaceNet (facenet-pytorch)
* ONNX Runtime
* Scikit-learn

---

# What Has Been Built So Far

## Environment Setup

A dedicated Python virtual environment was created to isolate dependencies.

Installed:

```bash
pip install insightface
pip install onnxruntime-gpu
pip install opencv-python==4.9.0.80
pip install numpy==1.26.4
pip install facenet-pytorch
pip install scikit-learn
```

GPU acceleration was attempted, but due to unresolved CUDA runtime dependency issues, the system currently runs on CPU.

Current performance:

```text
~3 FPS
```

---

## Webcam Input

The webcam feed is captured using:

```python
cap = cv2.VideoCapture(0)
```

This acts as the live input source.

During testing, the default camera index was not always reliable on this machine, so an optional `--camera-index` argument was added to let the app target a specific device when needed or simply leave it unset and attempt auto-detection.

---

## Face Detection

SCRFD detects faces and returns:

* Bounding boxes
* Facial landmarks
* Detection confidence

Example:

```text
Bounding Box → [x1, y1, x2, y2]
Confidence → 0.98
```

---

## Facial Landmark Detection

SCRFD provides:

* Right eye
* Left eye
* Nose
* Right mouth corner
* Left mouth corner

These are used to estimate face orientation.

---

## Face Alignment

Face alignment has been improved.

Earlier:

* Rotation was based on unstable angle computation.

Current:

* Correct angle calculation using eye geometry.
* Dynamic cropping after alignment.
* Fixed output size for consistency.

Angle:

```python
angle = np.degrees(np.arctan2(dy, dx))
```

Rotation:

```python
cv2.getRotationMatrix2D(center, angle, 1.0)
```

This improves consistency across embeddings.

---

## Blur Filtering

Before embedding generation, blur detection is performed.

Method:

```python
blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
```

Purpose:

* Reject blurry samples
* Improve database quality
* Reduce false matches

---

## Face Cropping

After alignment, the face is tightly cropped around the eye-center region.

This reduces unnecessary background.

---

## Face Preprocessing

FaceNet preprocessing has been corrected.

Current preprocessing:

* Resize to 160×160
* Convert BGR → RGB
* Convert to float32
* Prewhitening

Prewhitening:

```python
face = (face - mean) / std
```

This was an important fix because earlier embeddings had weak separation.

---

## Face Embedding Generation

FaceNet converts the processed face into a **512-dimensional embedding**.

Model:

```python
model = InceptionResnetV1(
    pretrained='vggface2'
).eval()
```

Output:

```text
torch.Size([1,512])
```

Changes:

* Added proper prewhitening
* Added L2 normalization

Normalization:

```python
embedding = F.normalize(embedding, p=2, dim=1)
```

This improves cosine similarity stability.

---

## Identity Registration

Identity registration now supports two modes.

### Webcam Registration

```bash
python main.py --register Alice
```

This captures live samples.

Current:

* Requires exactly one face
* Collects 10 valid samples
* Rejects blurry inputs

---

### Registration from Existing Images

```bash
python main.py --register_from_images Alice
```

Loads:

```text
data/images/Alice/*
```

This allows rebuilding embeddings after preprocessing changes.

Stored output:

```text
data/embeddings/Alice_embeddings.npy
```

Shape:

```text
(N,512)
```

---

## Face Recognition

Recognition mode:

```bash
python main.py --recognize Alice
```

Process:

```text
Live Face
   ↓
Embedding
   ↓
Centroid Matching
   ↓
Top-K Matching
   ↓
Combined Similarity Score
```

Changes:

Earlier:

* Direct max similarity matching

Current:

* Centroid matching
* Top-3 similarity averaging
* Combined final score

This improves robustness.

---

## CLI Integration

All operations are now unified into a single entrypoint:

```text
main.py
```

Supported modes:

```text
--register {name}
--register_from_images {name}
--recognize {name}
```

This makes the system modular and easier to deploy.

---

# Important Changes Made

## Alignment Fixes

Old:

* Incorrect angle logic
* Inconsistent rotations

New:

* Stable eye-angle alignment
* Dynamic recropping

---

## Embedding Fixes

Old:

* Only /255 normalization

New:

* Proper FaceNet prewhitening
* L2 normalized embeddings

This significantly improves separation.

---

## Database Matching Improvements

Old:

* Best-match only

New:

* Centroid + top-k combined scoring

Improves robustness.

---

## Registration Improvements

Old:

* 3 image samples

New:

* 10 webcam samples
* Existing image registration support
* Blur filtering

Improves database quality.

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

## Blur Detection

Filtering low-quality images.

---

## Tensors

Converting images into neural-network-compatible numerical data.

---

## Embeddings

Representing facial identity as a fixed-size vector.

---

## Gradient-Free Inference

Using:

```python
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

* Unknown face rejection threshold calibration
* Attendance logging
* Duplicate attendance prevention
* Performance optimization
* GPU acceleration
* 5-point alignment instead of 2-point eye alignment
* Liveness detection
* Database integration (Supabase / SQL)

---

# References

* SCRFD — Sample and Computation Redistribution for Efficient Face Detection
* RetinaFace — Single-stage Dense Face Localisation in the Wild
* FaceNet — A Unified Embedding for Face Recognition and Clustering

