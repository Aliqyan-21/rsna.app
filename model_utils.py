import os
import glob
import torch
import timm
import numpy as np
from PIL import Image
from collections import defaultdict
from torchvision import transforms
from typing import Dict, List, Union
import random

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default paths
YOLO_PT_AXIAL_T2_PATH = "models/axialY/best.pt"
SIAMESE_AXIAL_T2_REFIMG_ROOT_DIR = "ref_images/"
SIAMESE_AXIAL_T2_PT_LIST = sorted(glob.glob("models/axial/*.pth"))

LEVELS = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
CLASSES = ['Normal_Mild', 'Moderate', 'Severe']

# Transformations for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# DICOM Preprocessing
def convert_dicom_to_image(dcm_path):
    try:
        # DICOM file reading and conversion
        import pydicom
        from pydicom.pixel_data_handlers.util import apply_color_lut, apply_modality_lut, apply_voi_lut
        
        dicom = pydicom.dcmread(dcm_path)
        arr = dicom.pixel_array
        
        # Apply LUTs
        if dicom.PhotometricInterpretation == "PALETTE COLOR":
            arr = apply_color_lut(arr, dicom)
        else:
            arr = apply_modality_lut(arr, dicom)
            arr = apply_voi_lut(arr, dicom, index=0)
        
        # Handle MONOCHROME1
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            arr = np.amax(arr) - arr
        
        # Normalize and scale intensity
        lower, upper = np.percentile(arr, (1, 99))
        arr = np.clip(arr, lower, upper)
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        arr = (arr * 255).astype(np.uint8)
        
        # Convert grayscale to RGB
        if len(arr.shape) == 2:
            arr = np.stack([arr] * 3, axis=-1)
        
        return arr
    except Exception as e:
        print(f"Error processing DICOM file {dcm_path}: {e}")
        return None

# YOLO Patch Extraction
def extract_patch_with_yolo(image, yolo_model, detection_threshold=0.4):
    results = yolo_model(image)
    boxes = results[0].boxes  # Access detected boxes
    confidences = boxes.conf
    
    patches = []
    for i in range(len(confidences)):
        if confidences[i] > detection_threshold:
            x0, y0, x1, y1 = map(int, boxes.xyxy[i])
            patch = image[y0:y1, x0:x1]
            patches.append(patch)
    return patches

# Siamese Model
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

def siamese_base_inference(model: SiameseNetwork, test_image_tensor: torch.Tensor, confidence_threshold: float = 0.4):
    with torch.no_grad():
        class_scores = model(test_image_tensor)
    probabilities = {cls: torch.mean(scores).item() for cls, scores in class_scores.items()}
    total = sum(probabilities.values())
    normalized_probs = {cls: prob / total for cls, prob in probabilities.items()}

    max_prob = max(normalized_probs.values())
    if max_prob < confidence_threshold:
        return "Uncertain"
    return normalized_probs

# Load Models
def load_siamese_models(pt_path_list: List[str]) -> Dict[str, List[SiameseNetwork]]:
    reference_images = load_reference_images(SIAMESE_AXIAL_T2_REFIMG_ROOT_DIR, transform, DEVICE)
    models_by_level = defaultdict(list)
    for pt_path in pt_path_list:
        model = SiameseNetwork(pretrained=False)
        model.load_state_dict(torch.load(pt_path, map_location=DEVICE))
        model.eval().to(DEVICE)
        model.precompute_class_embeddings(reference_images)
        level = "_".join(os.path.basename(pt_path).split("_")[:2])
        models_by_level[level].append(model)
    return models_by_level

def load_reference_images(root_dir, transform, device, num_ref_images=5):
    random.seed(42)
    ref_images = defaultdict(list)
    for level in LEVELS:
        for cls in CLASSES:
            class_path = os.path.join(root_dir, level, cls)
            if os.path.exists(class_path):
                img_paths = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith('.png')]
                for img_path in random.sample(img_paths, min(len(img_paths), num_ref_images)):
                    img = Image.open(img_path).convert('RGB')
                    ref_images[cls].append(transform(img).to(device))
    return ref_images
