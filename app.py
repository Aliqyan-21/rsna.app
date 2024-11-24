from flask import Flask, render_template, request
import os
import pydicom
import numpy as np
from PIL import Image
import torch.nn as nn
import torch
from torchvision import transforms

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Model setup
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

model_path = "models/model.pth"
model = SimpleCNN(coord_size=2)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((312,312)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

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

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        ds = pydicom.dcmread(filepath)
        img_data = ds.pixel_array
        img = Image.fromarray(img_data.astype(np.uint8))

        img_tensor = transform(img).unsqueeze(0).to(device)

        # predicting the classes
        with torch.no_grad():
            outputs = model(img_tensor)  
            probs = torch.softmax(outputs, dim=1)

            normal_mild_prob = probs[0,0].item()
            moderate_prob = probs[0,1].item()
            severe_prob = probs[0,2].item()

            all_predictions.append({
                "file": file.filename,
                "normal_mild": normal_mild_prob,
                "moderate": moderate_prob,
                "severe": severe_prob
            })
        
    return render_template("predictions.html", predictions=all_predictions)

if __name__ == "__main__":
    app.run(debug=True)
