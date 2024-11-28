from flask import Flask, render_template, request
import os
import torch
from PIL import Image
from werkzeug.utils import secure_filename
import uuid
import logging
from ultralytics import YOLO
from model_utils import (
    convert_dicom_to_image,
    transform,
    extract_patch_with_yolo,
    siamese_base_inference,
    load_siamese_models,
)
import glob

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

logging.basicConfig(level=logging.INFO)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

YOLO_PT_AXIAL_T2_PATH = "models/axialY/best.pt"
detector = YOLO(YOLO_PT_AXIAL_T2_PATH)

SIAMESE_AXIAL_T2_PT_LIST = sorted(glob.glob("models/axial/*.pth"))
siamese_models = load_siamese_models(SIAMESE_AXIAL_T2_PT_LIST)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/dcm_upload')
def dcm_upload():
    return render_template("upload.html")

@app.route('/upload', methods=["POST"])
def upload_dcms():
    if 'files' not in request.files:
        return "No files uploaded"

    files = request.files.getlist('files')
    all_predictions = []

    if not files:
        return "No files selected"

    for file in files:
        if not file.filename:
            return "File name is empty or invalid"

        # Secure filename and save the file
        filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        logging.info(f"Processing file: {file.filename}")

        # Preprocess the DICOM file
        image = convert_dicom_to_image(filepath)
        if image is None:
            logging.warning(f"Failed to convert {file.filename} to image")
            all_predictions.append({"file": file.filename, "error": "Failed to process DICOM file."})
            continue

        # YOLO detection
        patches = extract_patch_with_yolo(image, detector)
        logging.info(f"Detected {len(patches)} patches for {file.filename}")
        if not patches:
            all_predictions.append({"file": file.filename, "error": "No patches detected."})
            continue

        # Siamese model predictions
        predictions = {"Normal_Mild": 0.0, "Moderate": 0.0, "Severe": 0.0}
        total_patches = 0

        for patch in patches:
            patch = Image.fromarray(patch)
            patch_tensor = transform(patch).unsqueeze(0).to(DEVICE)

            for _, models in siamese_models.items():
                for model in models:
                    probs = siamese_base_inference(model, patch_tensor)
                    
                    logging.info(f"Prediction for {file.filename} (patch): {probs}")
                    
                    if isinstance(probs, str) and probs == "Uncertain":
                        continue  # Skip this patch if uncertain
                    for cls, prob in probs.items():
                        predictions[cls] += prob

            total_patches += 1

        # If valid patches were processed, average predictions
        if total_patches > 0:
            predictions = {cls: round(prob / total_patches, 4) for cls, prob in predictions.items()}
            logging.info(f"Final predictions for {file.filename}: {predictions}")
        else:
            all_predictions.append({"file": file.filename, "error": "No valid patches processed."})
            continue

        all_predictions.append({
            "file": file.filename,
            "normal_mild": predictions["Normal_Mild"],
            "moderate": predictions["Moderate"],
            "severe": predictions["Severe"]
        })

    return render_template("predictions.html", predictions=all_predictions)

if __name__ == "__main__":
    app.run(debug=True)
