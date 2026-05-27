# Natural Disaster Intensity Analysis and Classification using AI

A deep learning project that classifies natural disaster types and analyzes their intensity from images using a Convolutional Neural Network (CNN) built with Keras and TensorFlow.

## Overview

Natural disasters cause devastating damage to ecosystems, infrastructure, and human lives. This project tackles the challenge of automatically detecting and classifying disaster types from images using a multilayered deep CNN model. The system captures real-time video frames via webcam, compares them against a pre-trained model, and displays the disaster type and intensity through an OpenCV window and Flask web interface.

## Disaster Categories

The model classifies images into 4 categories:
- 🌀 Cyclone
- 🏔️ Earthquake
- 🌊 Flood
- 🔥 Wildfire

## Pre-trained Model Download

> The `disaster.h5` model file (~9.4 MB) is hosted on Google Drive due to GitHub binary file limitations.

**[⬇️ Download disaster.h5 from Google Drive](https://drive.google.com/file/d/172tO9QiJ9W2DvhJUgB5AjNn0MXNVDF9D/view?usp=sharing)**

After downloading, place `disaster.h5` in the same folder as `app.py`.

## Model Architecture

| Layer | Output Shape | Parameters |
|---|---|---|
| Conv2D (32 filters, 3x3, ReLU) | (None, 62, 62, 32) | 896 |
| MaxPooling2D (2x2) | (None, 31, 31, 32) | 0 |
| Conv2D (32 filters, 3x3, ReLU) | (None, 29, 29, 32) | 9,248 |
| MaxPooling2D (2x2) | (None, 14, 14, 32) | 0 |
| Flatten | (None, 6272) | 0 |
| Dense (128, ReLU) | (None, 128) | 802,944 |
| Dense (4, Softmax) | (None, 4) | 516 |

**Total Parameters:** 813,604

## Training Results

- Dataset: 742 training images, 197 test images (4 classes)
- Image size: 64x64 RGB
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Epochs: 20

| Metric | Value |
|---|---|
| Final Training Accuracy | ~91.37% |
| Best Validation Accuracy | ~78.68% (Epoch 10) |

## Project Flow

1. User opens the integrated webcam via the Flask UI
2. Video frames are captured and analyzed by the pre-trained CNN model
3. Prediction is displayed on the UI and OpenCV window in real time

## Tech Stack

- **Python 3.9**
- **TensorFlow / Keras** - Model building and training
- **OpenCV** - Real-time webcam capture and display
- **Flask** - Web application framework
- **NumPy** - Array operations
- **ImageDataGenerator** - Data augmentation (rescale, shear, zoom, horizontal flip)

## Project Structure

```
Natural-Disaster-Intensity-Analysis/
|-- Model building/
|   |-- AI based Natural disaster analysis.ipynb   # Training notebook
|   |-- disaster.h5                                # Pre-trained model (see download link above)
|-- templates/
|   |-- home.html
|   |-- upload.html
|-- app.py                                         # Flask application
|-- requirements.txt
|-- README.md
```

## Setup and Usage

### Prerequisites

```bash
pip install -r requirements.txt
```

### Step 1 - Download the model

Download `disaster.h5` from the link above and place it in the root folder (same level as `app.py`).

### Step 2 - Run the Flask App

```bash
python app.py
```

Then open your browser at `http://localhost:5000`

### Running the Notebook (optional - to retrain)

1. Open `Model building/AI based Natural disaster analysis.ipynb` in Jupyter
2. Update the dataset path to your local directory
3. Run all cells to retrain the model

## Data Augmentation

Training images were augmented using Keras `ImageDataGenerator`:
- Rescaling (1/255)
- Shear range: 0.2
- Zoom range: 0.2
- Horizontal flip

## Notes

- Model was trained on Python 3.9 with TensorFlow/Keras 2.5.0
- Dataset images are not included due to size - organize your own images into `dataset/train_set/` and `dataset/test_set/` with subfolders for each class
