# Lung-Cancer-Detection-with-HR-SEMobileCapsNet

A lightweight deep learning application for detecting lung cancer using a custom hybrid model — **HR-SEMobileCapsNet** — which combines **Residual MobileNetV3** and **Capsule Networks**. This application is built using **FastAPI** and provides a simple web interface for image-based predictions.

---

## 🔍 Features

- Custom model: HR-SEMobileCapsNet (Residual MobileNetV3 + CapsuleNet)
- Pretrained weights for quick deployment
- FastAPI backend for fast and interactive model serving
- Simple HTML frontend for uploading lung scan images
- Lightweight and optimized for edge or web deployment

---

## ⚙️ Getting Started

### 1. Clone the Repository
Clone this GitHub repository and navigate into the project directory:
```
git clone https://github.com/Aakash67/Lung-Cancer-Detection-with-HR-SEMobileCapsNet.git
cd Lung-Cancer-Detection-with-HR-SEMobileCapsNet
```
### 2. (Optional) Create a Virtual Environment
To isolate dependencies, it's recommended to use a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
Install the required Python packages using pip:
pip install -r requirements.txt

### 4. Run the FastAPI App
Start the FastAPI server locally with:
```
uvicorn main:app --reload
```

### 5. Open in Browser
Open your web browser and navigate to: http://127.0.0.1:8000

---

##🧠 Model Overview
The HR-SEMobileCapsNet model is designed for medical image analysis, particularly lung cancer detection. It combines:

- Residual MobileNetV3: for fast and compact feature extraction

- Capsule Networks: for preserving spatial relationships and enhancing interpretability

This hybrid model is optimized for real-time performance and high accuracy, even on resource-constrained systems.

---

💻 Web Interface
A user-friendly web interface is included. Users can upload lung scan images (X-ray or CT), and the model processes the image and returns a diagnosis prediction instantly in the browser.
![image](https://github.com/user-attachments/assets/50627a2f-295b-4050-971a-f68b0f2ab72d)




