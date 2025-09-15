# ==============================================================
# 🚘 Vehicle & Number Plate Recognition App (Streamlit + YOLO)
# --------------------------------------------------------------
# This app performs two tasks:
#   1. Detect vehicles using YOLO pretrained COCO model
#   2. Detect license plates using a custom YOLO model
# Then it runs OCR (EasyOCR + Tesseract ensemble) on plates.
# It also shows cropped plates, downloads CSV/ZIP results.
# ==============================================================

import streamlit as st
import tempfile, os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import easyocr
import pytesseract   # fallback OCR
import re
import math

# ---------------- Streamlit UI setup ----------------
st.set_page_config(page_title="Vehicle & Number Plate Recognition", layout="wide")
st.title("🚘 Vehicle & Number Plate Recognition")

# ---------------- Load YOLO models ------------------
# Path to trained plate detection model
PLATE_MODEL = "runs/detect/train/weights/best.pt"   
# Vehicle detection uses YOLOv8 small pretrained on COCO
VEHICLE_MODEL = "yolov8s.pt"                        

plate_model = YOLO(PLATE_MODEL)
vehicle_model = YOLO(VEHICLE_MODEL)

# ---------------- OCR model setup -------------------
# EasyOCR for quick recognition, CPU mode
reader = easyocr.Reader(["en"], gpu=False)

# ---------------- Sidebar UI ------------------------
st.sidebar.header("Input")

# Extra sidebar options for plate handling
plates_within_vehicle_only = st.sidebar.checkbox(
    "Detect plates only inside detected vehicles", value=True)
download_plates_zip = st.sidebar.checkbox(
    "Enable download of plate crops (zip)", value=True)

# Choose source type
source_type = st.sidebar.radio("Source", ["Image", "Video"])
uploaded_file = st.sidebar.file_uploader("Upload image or video", 
                                         type=["jpg","jpeg","png","mp4","mov","avi"])

# Detection confidence threshold
conf_thresh = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.5, 0.05)
# OCR options
run_ocr = st.sidebar.checkbox("Run OCR on detected plates", value=True)
use_advanced_ocr = st.sidebar.checkbox("Use advanced OCR (ensemble)", value=True)

# ==============================================================
# ---------------- Helper Functions ----------------------------
# ==============================================================

def save_uploaded_temp(uploaded_file):
    """Save uploaded file to a temp location for YOLO inference."""
    ext = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name

def preprocess_crop(crop):
    """Sharpen + threshold plate crop for simple OCR (basic mode)."""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    sharpen = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    # Upscale if plate is too small
    h,w = sharpen.shape
    if w < 100:
        scale = 100 / w
        sharpen = cv2.resize(sharpen, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)

    # Otsu thresholding
    _, thr = cv2.threshold(sharpen,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thr

# ---------------- OCR helper mappings ----------------
# Mapping common OCR mistakes: O->0, I->1, S->5, etc.
CORRECT_MAP = {
    'O':'0','Q':'0','D':'0',
    'I':'1','L':'1','|':'1',
    'Z':'2',
    'S':'5','s':'5',
    'B':'8','G':'6'
}
WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"

def clean_text_for_plate(t: str) -> str:
    """Clean OCR output to look like license plate text."""
    if not t: return ""
    s = re.sub(r'[^A-Za-z0-9\-]', '', t.upper())
    s = "".join(CORRECT_MAP.get(ch, ch) for ch in s)
    s = re.sub(r'^[^A-Z0-9]+|[^A-Z0-9]+$', '', s)
    return s

# Several advanced preprocessing functions follow
# (deskew, upscale+CLAHE, adaptive threshold + morphology).
# These are used in "advanced OCR ensemble" mode.

def deskew_crop(crop_bgr): ...
def upscale_and_clahe(crop_bgr, min_w=220): ...
def adaptive_and_morph(prep): ...
def tesseract_read(img, psm=7): ...

def run_ocr_on_crop(crop_rgb, use_advanced=True):
    """
    Run OCR on a plate crop.
    - Basic mode: simple preprocess + EasyOCR + Tesseract fallback
    - Advanced mode: multiple preprocessing variants + ensemble scoring
    """
    # (Implementation same as your code, with comments inside)
    ...

# ---------------- Drawing function -------------------
def draw_boxes_on_image(image, detections, plate_texts):
    """Draw bounding boxes with labels / OCR text on image."""
    img = image.copy()
    for d in detections:
        x1,y1,x2,y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        label = d.get("label", "")
        conf = d.get("conf", 0)
        color = (0,255,0)

        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)

        # Show OCR text if available, else show label + confidence
        text = plate_texts.get((x1,y1,x2,y2), f"{label} {conf:.2f}")
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, max(0,y1-th-8)), (x1+tw+8, y1), color, -1)
        cv2.putText(img, text, (x1+4, max(16, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

    return img

# ==============================================================
# ---------------- Main Application Flow ------------------------
# ==============================================================

if uploaded_file is not None:
    tmp_path = save_uploaded_temp(uploaded_file)
    st.write(f"Processing: {uploaded_file.name}")

    if source_type == "Image":
        # -------- 1. Load image --------
        image = np.array(Image.open(tmp_path).convert("RGB"))

        # -------- 2. Vehicle detection --------
        v_results = vehicle_model.predict(source=tmp_path, conf=conf_thresh, save=False)
        v_dets = []
        for r in v_results:
            for box in r.boxes or []:
                x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = vehicle_model.names.get(cls_id, str(cls_id))
                # Keep only relevant classes
                if cls_name.lower() in ["car", "truck", "bus", "motorbike", "motorcycle", "bicycle"]:
                    v_dets.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"label":cls_name,"conf":conf})

        # -------- 3. Plate detection --------
        p_results = plate_model.predict(source=tmp_path, conf=conf_thresh, save=False)
        plate_dets = []
        for r in p_results:
            for box in r.boxes or []:
                x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = plate_model.names.get(cls_id, str(cls_id))
                plate_dets.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"label":cls_name,"conf":conf})

        # -------- 4. Associate plates with vehicles --------
        # (center containment + IoU fallback)
        def iou(boxA, boxB): ...
        plate_assigned = {}
        for pi, pd in enumerate(plate_dets): ...
        if plates_within_vehicle_only and v_dets:
            plate_dets = [pd for i,pd in enumerate(plate_dets) if plate_assigned.get(i) is not None]

        # -------- 5. Run OCR on detected plates --------
        # Collect crops, run OCR, clean results
        plate_crops, plate_texts, ocr_rows = [], {}, []
        for i, pd in enumerate(plate_dets): ...
            # crop + OCR + save text

        # -------- 6. Annotate image + show results --------
        annotated = draw_boxes_on_image(image, v_dets + plate_dets, plate_texts)
        st.image(annotated, channels="RGB", use_container_width=True)

        # Show cropped plates
        if plate_crops: ...
        # Allow ZIP download of crops
        if download_plates_zip and plate_crops: ...
        # Show OCR results in table and CSV
        if ocr_rows: ...

    else:
        # -------- Video mode --------
        st.info("Video inference: results saved to runs/detect/predict")
        _ = vehicle_model.predict(source=tmp_path, conf=conf_thresh, save=True)
        _ = plate_model.predict(source=tmp_path, conf=conf_thresh, save=True)
        st.success("Processing complete. Check runs/detect/predict for outputs.")
        st.video(tmp_path)

    # Cleanup temporary file
    try: os.remove(tmp_path)
    except: pass
else:
    st.info("Upload an image or video to start detection.")
