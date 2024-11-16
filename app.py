from flask import Flask, render_template, request
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/upload', methods=["POST"])
def upload_dic():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    if not file.filename:
        return "No file selected"

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filenamej)
        file.save(filepath)

    return f"File uploaded: {file.filename}" 

if __name__ == "__main__":
    app.run(debug=True)
