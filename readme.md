# Vehicle & License Plate Detection App 🚗🔍

A **Streamlit web app** that uses YOLO models to detect vehicles and license plates in images or video, predict vehicle type, and optionally run OCR on license plates.  
This project combines **vehicle detection, plate detection, and OCR** into a single easy-to-use interface.

---

## ✨ Features

- **Vehicle detection** with YOLO (cars, bikes, buses, trucks, etc.)
- **License plate detection** with YOLO plate model
- Option to **restrict plate detection to inside detected vehicles**
- **OCR support** for license plates (basic & advanced modes)
- Annotated output images with bounding boxes
- Display of **plate crops** as thumbnails
- Download:
  - Plate crops as a **ZIP archive**
  - OCR results as a **CSV file**
  - 
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

  





