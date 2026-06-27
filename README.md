# Real-Time Face Recognition System

A real-time face recognition pipeline built using **SCRFD (InsightFace)**, **OpenCV**, and **FaceNet**.

This project detects faces, aligns them, generates embeddings, stores known identities, and compares live faces for recognition.

The system supports:

* Face registration from webcam
* Face registration from existing images
* Real-time face verification
* Blur filtering for cleaner embeddings
* Embedding comparison using centroid + top-k cosine similarity

---

# Project Structure

```text
face_recognition/
│── src/
│   │── main.py
│   │── detect.py
│   │── align.py
│   │── embed.py
│   │── recognize.py
│   │── register.py
│── journey.md
│
├── data/
│   ├── images/
│   │   ├── person1/
│   │   ├── person2/
│   │
│   ├── embeddings/
│       ├── person1_embeddings.npy
│       ├── person2_embeddings.npy
```

---

# Installation

Create virtual environment:

```bash
python -m venv venv
```

Activate:

Windows:

```bash
venv\Scripts\activate
```

Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install insightface
pip install onnxruntime-gpu
pip install opencv-python==4.9.0.80
pip install numpy==1.26.4
pip install facenet-pytorch
pip install scikit-learn
```

---

# How to Use

All operations are done through:

```bash
python main.py
```

Optional camera selection:

```bash
python main.py --recognize Alice --camera-index 0
```

If `--camera-index` is omitted, the app will try to auto-detect a usable webcam. This is helpful when multiple cameras are available or when the default index is not working.

---

# 1. Register From Webcam

Registers a person using live webcam samples.

Command:

```bash
python main.py --register Alice
```

If needed, you can also specify a camera index:

```bash
python main.py --register Alice --camera-index 0
```

What it does:

* Opens webcam
* Collects 10 face samples
* Detects and aligns face
* Generates embeddings
* Saves embeddings

Output:

```text
data/embeddings/Alice_embeddings.npy
```

Requirements:

* Only one face in frame
* Good lighting
* Minimal blur

---

# 2. Register From Existing Images

Registers a person from images stored locally.

Place images inside:

```text
data/images/Alice/
```

Example:

```text
data/images/Alice/img1.jpg
data/images/Alice/img2.jpg
data/images/Alice/img3.jpg
```

Run:

```bash
python main.py --register_from_images Alice
```

Output:

```text
data/embeddings/Alice_embeddings.npy
```

Use this when:

* rebuilding embeddings
* testing preprocessing changes
* bulk registration

---

# 3. Recognize / Verify

Compares live webcam face against stored embeddings.

Run:

```bash
python main.py --recognize Alice
```

Output:

```text
Alice: 84.25%
```

If similarity is below threshold:

```text
Unknown
```

---

# Pipeline

```text
Input Image / Webcam
        ↓
Face Detection (SCRFD)
        ↓
Landmark Extraction
        ↓
Face Alignment
        ↓
Blur Filtering
        ↓
Preprocessing
        ↓
FaceNet Embedding
        ↓
Database Matching
        ↓
Recognition
```

---

# Notes

* Registration quality directly affects recognition accuracy.
* Re-register identities after changing preprocessing or alignment logic.
* Current implementation supports **single-person verification**.

For detailed implementation journey, experiments, and improvements:

See:

```text
journey.md
```

---

# References

* SCRFD — Sample and Computation Redistribution for Efficient Face Detection
* RetinaFace — Single-stage Dense Face Localisation in the Wild
* FaceNet — A Unified Embedding for Face Recognition and Clustering
