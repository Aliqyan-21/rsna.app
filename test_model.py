import os
import glob
import torch
import timm
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_color_lut, apply_modality_lut, apply_voi_lut
from PIL import Image
from collections import defaultdict
from torchvision import transforms
from ultralytics import YOLO
from tqdm import tqdm
from typing import Dict, List, Union
import random

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths setup (updated from second.py)
YOLO_PT_AXIAL_T2_PATH = "models/axialY/best.pt"  # YOLO model path
SIAMESE_AXIAL_T2_REFIMG_ROOT_DIR = "ref_images/"  # Reference images root
SIAMESE_AXIAL_T2_PT_LIST = sorted(glob.glob("models/axial/*.pth"))  # Siamese models

LEVELS = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
CLASSES = ['Normal_Mild', 'Moderate', 'Severe']

# YOLO confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# Load YOLO model for detection
detector = YOLO(YOLO_PT_AXIAL_T2_PATH)

# Transformations for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def convert_dicom_to_image(dcm_path):
    try:
        # Read the DICOM file
        dicom = pydicom.dcmread(dcm_path)
        arr = dicom.pixel_array

        # Apply LUTs if necessary
        if dicom.PhotometricInterpretation == "PALETTE COLOR":
            arr = apply_color_lut(arr, dicom)
        else:
            arr = apply_modality_lut(arr, dicom)
            arr = apply_voi_lut(arr, dicom, index=0)

        # Handle MONOCHROME1 interpretation
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            arr = np.amax(arr) - arr  # Invert pixel values

        # Intensity normalization using percentiles
        lower, upper = np.percentile(arr, (1, 99))  # Exclude extreme outliers
        arr = np.clip(arr, lower, upper)
        arr = arr - np.min(arr)
        arr = arr / np.max(arr)  # Normalize to [0, 1]
        arr = (arr * 255).astype(np.uint8)  # Scale to [0, 255]

        # Convert grayscale to RGB if needed
        if len(arr.shape) == 2:  # Single channel
            arr = np.stack([arr] * 3, axis=-1)

        return arr

    except Exception as e:
        print(f"Error processing DICOM file {dcm_path}: {e}")
        return np.zeros((224, 224, 3), dtype=np.uint8)

def extract_patch_with_yolo(image: np.ndarray):
    """Detect and extract patch using YOLO."""
    results = detector.predict(source=image, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()

    if len(detections) == 0:
        print("No detections found.")
        return None

    # Get the highest-confidence detection
    best_detection = max(detections, key=lambda x: x[4])  # Sort by confidence
    x0, y0, x1, y1, confidence, class_id = best_detection[:6]

    if confidence < CONFIDENCE_THRESHOLD:
        print("Detection confidence too low.")
        return None

    # Extract and return the patch
    patch = image[int(y0):int(y1), int(x0):int(x1)]
    return patch

def load_reference_images(root_dir, transform, device, num_ref_images=5):
    random.seed(42)

    ref_images = defaultdict(list)

    for level in LEVELS:
        for cls in CLASSES:
            class_path = os.path.join(root_dir, level, cls)
            if not os.path.exists(class_path) or not os.path.isdir(class_path):
                print(f"Warning: Class path {class_path} does not exist or is not a directory.")
                continue

            # List all PNG files in the directory
            img_paths = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith('.png')]
            
            if not img_paths:
                print(f"Warning: No images found in {class_path}")
                continue

            # Select random samples (up to `num_ref_images`)
            num_samples = min(len(img_paths), num_ref_images)
            samples_of_class = random.sample(img_paths, num_samples)

            # Load and transform images
            for img_path in samples_of_class:
                try:
                    img = Image.open(img_path).convert('RGB')  # Open image
                    transformed_img = transform(img).to(device)  # Transform and move to device
                    ref_images[cls].append(transformed_img)  # Append to flat dictionary
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    return ref_images

class SiameseNetwork(torch.nn.Module):
    def __init__(self, model_name: str = 'resnet18', pretrained: bool = True):
        super(SiameseNetwork, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = torch.nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = torch.nn.Identity()
        self.similarity = torch.nn.CosineSimilarity(dim=1)
        self.class_embeddings: Union[Dict[str, torch.Tensor], None] = None

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def precompute_class_embeddings(self, reference_images: Dict[str, List[torch.Tensor]]):
        self.class_embeddings = {}
        for class_label, images in reference_images.items():
            with torch.no_grad():
                embeddings = [self.forward_once(img.unsqueeze(0)) for img in images]
                self.class_embeddings[class_label] = torch.stack(embeddings)

    def forward(self, img1, img2=None):
        if img2 is not None:
            embedding1 = self.forward_once(img1)
            embedding2 = self.forward_once(img2)
            return self.similarity(embedding1, embedding2)
        else:
            embedding1 = self.forward_once(img1)
            similarities = {}
            for class_label, embeddings in self.class_embeddings.items():
                similarities[class_label] = self.similarity(
                    embedding1.unsqueeze(1), embeddings
                ).mean(dim=1)
            return similarities

def siamese_base_inference(model: SiameseNetwork, test_image_tensor: torch.Tensor, confidence_threshold: float = 0.5):
    with torch.no_grad():
        class_scores = model(test_image_tensor)
    probabilities = {cls: torch.mean(scores).item() for cls, scores in class_scores.items()}
    total = sum(probabilities.values())
    normalized_probs = {cls: prob / total for cls, prob in probabilities.items()}

    max_prob = max(normalized_probs.values())
    if max_prob < confidence_threshold:
        return "Uncertain"
    return normalized_probs

def load_siamese_models(pt_path_list: list) -> Dict[str, List[SiameseNetwork]]:
    reference_images = load_reference_images(
        SIAMESE_AXIAL_T2_REFIMG_ROOT_DIR, transform, DEVICE
    )

    models_by_level = defaultdict(list)
    for pt_path in tqdm(pt_path_list, desc="Loading Siamese Models"):
        model = SiameseNetwork(pretrained=False)
        model.load_state_dict(torch.load(pt_path, map_location=DEVICE))
        model.eval().to(DEVICE)
        model.precompute_class_embeddings(reference_images)  # Pass flat dictionary
        level = "_".join(os.path.basename(pt_path).split("_")[:2])
        models_by_level[level].append(model)

    return models_by_level

dcm_files = [
    "test_image/mild/1.dcm",
    "test_image/mild/2.dcm",
    "test_image/mild/3.dcm",
    "test_image/mild/4.dcm",
    "test_image/mild/5.dcm",
    "test_image/moderate/1.dcm",
    "test_image/moderate/2.dcm",
    "test_image/moderate/3.dcm",
    "test_image/moderate/4.dcm",
    "test_image/moderate/5.dcm",
    "test_image/severe/1.dcm",
    "test_image/severe/2.dcm",
    "test_image/severe/3.dcm",
    "test_image/severe/4.dcm",
    "test_image/severe/5.dcm",
]

siamese_models = load_siamese_models(SIAMESE_AXIAL_T2_PT_LIST)

for dcm_file in dcm_files:
    print(f"Processing {dcm_file}...")
    dicom_image = convert_dicom_to_image(dcm_file)
    patch = extract_patch_with_yolo(dicom_image)

    if patch is None:
        print(f"No valid patch found for {dcm_file}, skipping.")
        continue

    patch_tensor = transform(Image.fromarray(patch)).unsqueeze(0).to(DEVICE)

    combined_probabilities = defaultdict(list)
    for model in siamese_models["l1_l2"]:  # Replace with relevant level
        probabilities = siamese_base_inference(model, patch_tensor)
        for cls, prob in probabilities.items():
            combined_probabilities[cls].append(prob)

    final_probs = {cls: np.mean(probs) for cls, probs in combined_probabilities.items()}
    print(f"Predicted Class for {dcm_file}: {max(final_probs, key=final_probs.get)}")
