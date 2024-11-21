from flask import Flask, render_template, request
import os
import pydicom
import numpy as np
from PIL import Image
import torch.nn as nn
import torch

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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

        x = x.view(x.size(0), -1)  # Flatten

        if coords is not None:
            x = torch.cat((x, coords), dim=1)  # Concatenate with coordinates
            x = self.relu(self.fc1_with_coords(x))
        else:
            x = self.relu(self.fc1_without_coords(x))

        x = self.fc2(x)
        return x

model_path = "models/model.pth"
model = SimpleCNN(coord_size=2)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

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

    if not files:
        return "No files selected"

    for file in files:
        # Ensure filename is not empty or None
        if not file.filename:
            return "File name is empty or invalid"

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        ds = pydicom.dcmread(filepath)
        img = ds.pixel_array.astype(float)

        img -= np.min(img)
        img /= np.max(img)

        img = (img*255).astype(np.uint8)

        # Apply preprocessing here...
        # ...

        # Predict outcome here...
        # ...

    # placeholders we need to send the predictions to the prediction webpage for showing prediction later when we have the predictions
        return "saved"
    else:
        return "file not found"


if __name__ == "__main__":
    app.run(debug=True)
