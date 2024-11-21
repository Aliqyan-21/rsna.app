import torch
import pydicom
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SimpleCNN class similiar to baseline model
class SimpleCNN(nn.Module):
    def __init__(self, coord_size=None):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.flattened_size = 128 * 39 * 39
        self.coord_size = coord_size if coord_size is not None else 0
        
        self.fc1_with_coords = nn.Linear(self.flattened_size + self.coord_size, 256)
        self.fc1_without_coords = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x, coords=None):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)  

        if coords is not None:
            x = torch.cat((x, coords), dim=1)  
            x = self.relu(self.fc1_with_coords(x))
        else:
            x = self.relu(self.fc1_without_coords(x))

        x = self.fc2(x)
        return x

# model loading
model = SimpleCNN(coord_size=2)
state_dict = torch.load('models/model.pth', map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# similar to DataLoader usage in notebook
class DICOMDataset(Dataset):
    def __init__(self, dicom_paths, coords=None, transform=None):
        self.dicom_paths = dicom_paths
        self.coords = coords
        self.transform = transform

    def __len__(self):
        return len(self.dicom_paths)

    def __getitem__(self, idx):
        dicom_path = self.dicom_paths[idx]
        image = self.load_dicom_image(dicom_path)
        
        if self.transform:
            image = self.transform(image)
        
        if self.coords is not None:
            return image, self.coords
        else:
            return image, None

    def load_dicom_image(self, dcm_path):
        dicom_data = pydicom.dcmread(dcm_path)
        image_data = dicom_data.pixel_array
        image = Image.fromarray(image_data.astype(np.uint8))
        return image

# dicom images transforming (preprocessing)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((312, 312)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# array of dicom images
dicom_paths = ["10.dcm"]  

# Dicom dataset and DataLoader
dataset = DICOMDataset(dicom_paths, coords=torch.tensor([1.0, 2.0]), transform=transform)
testloader = DataLoader(dataset, batch_size=1, shuffle=False)

# predicions done like this
def predict_test_data(testloader):
    normal_mild_probs = []
    moderate_probs = []
    severe_probs = []
    predictions = []
    
    model.eval()  
    
    with torch.no_grad():  
        for images, coords in tqdm(testloader):
            images = images.to(device)
            
            outputs = model(images, coords) if coords is not None else model(images, None)
            
            probs = torch.softmax(outputs, dim=1)
            
            # storing each class confidence/probabilities
            normal_mild_probs.append(probs[0, 0].item())
            moderate_probs.append(probs[0, 1].item())
            severe_probs.append(probs[0, 2].item())
            predictions.append(probs)

    return normal_mild_probs, moderate_probs, severe_probs, predictions

# fingers crossed
normal_mild_probs, moderate_probs, severe_probs, predictions = predict_test_data(testloader)
print(f"Normal/Mild Probabilities: {normal_mild_probs}")
print(f"Moderate Probabilities: {moderate_probs}")
print(f"Severe Probabilities: {severe_probs}")
