# Vehicle & License Plate Detection App 🚗🔍

A **Streamlit web app** that uses YOLO models to detect vehicles and license plates in images or video, predict vehicle type, and optionally run OCR on license plates.  
This project combines **vehicle detection, plate detection, and OCR** into a single easy-to-use interface.
YOLO is an object detection algorithm that processes the image in a single pass (one look) instead of scanning multiple regions like older methods (e.g., R-CNN).

“You Only Look Once” → The model looks at the image once and directly predicts:

Bounding boxes (where objects are)

Class probabilities (what the objects are)

👉 This makes YOLO fast and real-time, which is why it’s widely used in tasks like license plate detection, surveillance, and self-driving cars.

---

## ✨ Features

- **Vehicle detection** with YOLO (cars, bikes, buses, trucks, etc.)
- **License plate detection** with YOLO plate model
- Option to **restrict plate detection to inside detected vehicles**
- **OCR support** for license plates (basic & advanced modes)
- Annotated output images with bounding boxes
- Display of **plate crops** as thumbnails

# 🚗 License Plate Detection using YOLOv8

## 1. Problem Statement
- Automatic license plate detection is crucial for:
  - Traffic monitoring
  - Toll collection
  - Security & law enforcement
- Challenge:
  - Vehicle detection works well.
  - License plates are **small** and **harder to detect**.

---

## 2. Dataset
- Source: **Roboflow License Plate dataset** (~10k images).
- Contains annotated license plates.
- Train / Validation / Test split.
  
**Dataset visualization (`labels.jpg`):**
- Top-left: Number of instances per class.
- Bottom: Plate positions (x, y) across images.
- Right: Plate size distribution (width, height).
- Observation: Plates are **tiny objects** compared to the image.

---

## 3. Model & Training
- **Model:** YOLOv8 Nano (`yolov8n.pt`)
- **Training:**
  - Epochs: 50
  - Input size: 640×640
  - Batch size reduced due to memory
- **Why YOLOv8n?**
  - Lightweight, fast, good for demo/prototype.

**Training curves (`results.png`):**
- **Losses (box, cls, dfl):** Decreasing → model is learning.
- **Precision ~0.9:** Predictions are correct when detected.
- **Recall ~0.5:** Still misses some plates.
- **mAP50 ~0.55, mAP50-95 ~0.35:** Good baseline for small objects.

---

## 4. Results
- Model can detect license plates in test images.
- Saves **annotated images** + **labels**.
- Example detections:
  - Visible plates detected well.
  - Struggles in low-light, blur, or tilted angles.

---

## 5. Conclusion & Next Steps
✅ Achieved license plate detection with YOLOv8.  
✅ Good **precision**, moderate **recall**.  
✅ Suitable as a baseline detector.

  

# Inference



---

## 1. Losses (box, cls, dfl)
- **Box loss (Localization loss):** Measures how well predicted bounding boxes align with ground-truth plates.  
- **Cls loss (Classification loss):** Measures how well the model predicts the correct class (e.g., "license plate").  
- **Dfl loss (Distribution Focal Loss):** Refines bounding box edges for better accuracy.  

✅ Losses **decreasing** → Model is learning correctly.

---

## 2. Precision (~0.9)
- **Definition:** Of all detected plates, how many are actually correct.  
- **Interpretation:** 90% of detected plates are correct (few false positives).  
👉 Model is good at **avoiding wrong detections**.

---

## 3. Recall (~0.5)
- **Definition:** Of all real plates present, how many the model actually detected.  
- **Interpretation:** The model is catching only ~50% of plates.  
👉 Model is **missing many plates** (low sensitivity).

---

## 4. mAP (Mean Average Precision)
- **AP (Average Precision):** Area under the Precision-Recall curve.  
- **mAP:** Mean AP across all classes (in this case, only "plate").  
- Evaluated at different IoU thresholds:
  - **mAP@50 (~0.55):** At IoU ≥ 0.5, the model detects plates correctly ~55% of the time.  
  - **mAP@50-95 (~0.35):** Stricter metric (IoU 0.5 → 0.95). Performance drops when precise localization is required.  

👉 Indicates decent baseline performance but room for improvement.

---

## 5. Quick Summary
- **Losses ↓** → Model is learning.  
- **Precision (0.9):** Very few wrong detections.  
- **Recall (0.5):** Still missing ~half the plates.  
- **mAP50 (0.55):** Decent starting point.  
- **mAP50-95 (0.35):** Struggles with precise localization.  

---




