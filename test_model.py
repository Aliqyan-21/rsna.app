import os
import glob
import torch
import timm
import numpy as np
import pydicom
from PIL import Image
from collections import defaultdict
from torchvision import transforms
from ultralytics import YOLO
from tqdm import tqdm
from typing import Dict, List, Union

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

def convert_dicom_to_image(dcm_path: str) -> np.ndarray:
    """Convert DICOM file to normalized RGB image."""
    try:
        dicom = pydicom.dcmread(dcm_path)
        arr = dicom.pixel_array.astype(float)
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) * 255.0
        if len(arr.shape) == 2:  # Grayscale to RGB
            arr = np.stack([arr] * 3, axis=-1)
        return arr.astype(np.uint8)
    except Exception as e:
        print(f"Error reading DICOM file {dcm_path}: {e}")
        return np.zeros((224, 224, 3), dtype=np.uint8)

def load_reference_images(root_dir: str, transform: transforms.Compose, device: torch.device, num_ref_images=40) -> Dict[str, List[torch.Tensor]]:
    """Load and preprocess reference images."""
    ref_images_per_class = defaultdict(list)
    for level in sorted(os.listdir(root_dir)):
        level_path = os.path.join(root_dir, level)
        if not os.path.isdir(level_path):
            continue
        for cls in sorted(os.listdir(level_path)):
            class_path = os.path.join(level_path, cls)
            if not os.path.isdir(class_path):
                continue
            image_paths = glob.glob(os.path.join(class_path, "*.png"))[:num_ref_images]
            images = [transform(Image.open(img).convert("RGB")).to(device) for img in image_paths]
            ref_images_per_class[f"{level}/{cls}"] = images
    return ref_images_per_class

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
        model.precompute_class_embeddings(reference_images)
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
    dicom_tensor = transform(Image.fromarray(dicom_image)).unsqueeze(0).to(DEVICE)

    combined_probabilities = defaultdict(list)
    for model in siamese_models["l1_l2"]:  # Replace with relevant level
        probabilities = siamese_base_inference(model, dicom_tensor)
        for cls, prob in probabilities.items():
            combined_probabilities[cls].append(prob)

    final_probs = {cls: np.mean(probs) for cls, probs in combined_probabilities.items()}
    print(f"Predicted Class for {dcm_file}: {max(final_probs, key=final_probs.get)}")
