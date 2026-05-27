from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load pre-trained model
MODEL_PATH = 'disaster.h5'
model = load_model(MODEL_PATH)

# Class labels
CLASSES = ['Cyclone', 'Earthquake', 'Flood', 'Wildfire']

# Class descriptions
CLASS_INFO = {
    'Cyclone': {
        'icon': '🌀',
        'color': '#3b82f6',
        'description': 'A powerful rotating storm system with strong winds and heavy rainfall.',
        'safety': 'Evacuate coastal areas, seek sturdy shelter, avoid windows.'
    },
    'Earthquake': {
        'icon': '🏔️',
        'color': '#f59e0b',
        'description': 'Sudden shaking of the ground caused by seismic activity.',
        'safety': 'Drop, cover, and hold on. Stay away from windows and heavy objects.'
    },
    'Flood': {
        'icon': '🌊',
        'color': '#06b6d4',
        'description': 'Overflow of water submerging land that is usually dry.',
        'safety': 'Move to higher ground immediately. Do not walk through floodwaters.'
    },
    'Wildfire': {
        'icon': '🔥',
        'color': '#ef4444',
        'description': 'Uncontrolled fire spreading rapidly through vegetation.',
        'safety': 'Evacuate immediately. Close all windows and doors before leaving.'
    }
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_disaster(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]
    pred_index = np.argmax(preds)
    pred_class = CLASSES[pred_index]
    confidence = float(preds[pred_index]) * 100
    all_probs = {CLASSES[i]: round(float(preds[i]) * 100, 2) for i in range(len(CLASSES))}
    return pred_class, confidence, all_probs

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('upload.html')

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF, WEBP).'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    pred_class, confidence, all_probs = predict_disaster(filepath)
    info = CLASS_INFO[pred_class]

    # Convert image to base64 for display
    with Image.open(filepath) as img:
        img = img.convert('RGB')
        buffered = BytesIO()
        img.save(buffered, format='JPEG')
        img_b64 = base64.b64encode(buffered.getvalue()).decode()

    return jsonify({
        'prediction': pred_class,
        'confidence': round(confidence, 2),
        'all_probabilities': all_probs,
        'icon': info['icon'],
        'color': info['color'],
        'description': info['description'],
        'safety': info['safety'],
        'image': img_b64
    })

if __name__ == '__main__':
    app.run(debug=True)
