# Real-Time Face Recognition System

A real-time face recognition pipeline built using **SCRFD (InsightFace)** and **OpenCV**, designed for multi-person recognition, face alignment, and embedding-based identification.

This project focuses on building the full computer vision pipeline step-by-step, starting from face detection and progressing toward recognition, database matching, and attendance logging.

---

# Project Objective

The goal of this project is to build a complete face recognition system capable of:

* Detecting multiple faces in real time
* Extracting facial landmarks
* Aligning faces for consistent orientation
* Generating facial embeddings
* Matching identities using similarity metrics
* Logging recognized individuals

The project emphasizes both **implementation** and **understanding of the underlying concepts**.

---

# Current Pipeline

```text
Webcam Feed
     ↓
SCRFD Face Detection
     ↓
Bounding Box Extraction
     ↓
Facial Landmark Detection
     ↓
Face Alignment (In Progress)
     ↓
Face Embedding Generation
     ↓
Face Recognition
     ↓
Attendance Logging
```

---

# Technologies Used

* Python 3.10
* OpenCV
* InsightFace
* SCRFD
* NumPy
* ONNX Runtime
* PyTorch (for future embedding generation)

---

# Work Completed

## 1. Environment Setup

A dedicated Python virtual environment was created for dependency isolation.

Installed:

```bash
pip install insightface
pip install onnxruntime-gpu
pip install opencv-python
pip install numpy
pip install torch torchvision torchaudio
```

GPU acceleration through ONNX Runtime was attempted, but due to unresolved CUDA runtime dependency issues, the current implementation uses CPU inference.

---

## 2. Webcam Integration

A real-time webcam stream was established using OpenCV:

```python
cap = cv2.VideoCapture(0)
```

This provides continuous frame capture for live face detection.

---

## 3. Face Detection using SCRFD

The SCRFD detector from InsightFace was initialized:

```python
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
```

The detector processes each frame and returns:

* Bounding box coordinates
* Facial landmarks
* Detection confidence score

---

## 4. Bounding Box Visualization

Each detected face is enclosed using:

```python
cv2.rectangle()
```

Bounding boxes help localize faces spatially in the frame.

Returned format:

```text
[x1, y1, x2, y2]
```

Where:

* `(x1, y1)` → top-left corner
* `(x2, y2)` → bottom-right corner

---

## 5. Detection Confidence

Each face includes a confidence score:

```python
face.det_score
```

This indicates how certain the detector is about the face prediction.

Displayed on-screen above the bounding box.

---

## 6. Multi-Face Detection

The detector currently supports multiple simultaneous faces.

Each detected face is processed independently.

Current loop structure:

```python
for face in faces:
```

This makes the system scalable for multi-person recognition.

---

## 7. FPS Measurement

Performance is measured using:

```python
fps = 1 / (curr_time - prev_time)
```

To stabilize noisy readings, an exponential moving average is used:

```python
fps = 0.9 * fps + 0.1 * instant_fps
```

This smooths sudden fluctuations in frame timing.

Current performance:

```text
~3 FPS (CPU)
```

---

## 8. Facial Landmark Extraction

SCRFD returns five key facial landmarks:

```text
Left Eye
Right Eye
Nose
Mouth Left
Mouth Right
```

These are stored as:

```python
face.kps
```

Shape:

```text
(5,2)
```

Each point contains:

```text
[x, y]
```

coordinates.

---

## 9. Landmark Visualization

Each landmark is drawn using:

```python
cv2.circle()
```

This allows direct visualization of facial structure and verifies detector consistency.

Landmarks remain attached to the face while moving.

---

# Concepts Learned So Far

## Face Detection

Detecting facial regions within an image.

---

## Bounding Box Geometry

Understanding image coordinates:

```text
(x, y)
```

for object localization.

---

## Facial Landmark Detection

Keypoint extraction for important facial regions.

---

## Confidence Scores

Understanding model certainty in predictions.

---

## Real-Time Inference

Running detection continuously on live webcam frames.

---

## FPS Smoothing

Using Exponential Moving Average (EMA) for stable performance estimation.

---

# Current Focus

The current task is implementing **face alignment**.

This involves:

1. Extracting eye coordinates
2. Computing the angle between the eyes
3. Rotating the image so the eyes become horizontal

This improves consistency for the face embedding stage.

---

# Planned Features

* Face alignment
* Face cropping
* Face embedding generation using FaceNet
* Identity database creation
* Cosine similarity matching
* Unknown face rejection
* Multi-person recognition
* Attendance CSV logging
* Recognition benchmarking
* Accuracy evaluation

---

# References

### SCRFD Paper

Sample and Computation Redistribution for Efficient Face Detection

---

### RetinaFace Paper

Single-stage Dense Face Localisation in the Wild

---

### FaceNet Paper

A Unified Embedding for Face Recognition and Clustering

---

# Future Goal

The final system aims to become a deployable real-time recognition pipeline suitable for:

* Smart attendance systems