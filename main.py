from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import torch
from torchvision import transforms
from PIL import Image
import io
import os

from cv_model.network import load_model  # ✅ Updated import path

# Initialize FastAPI and templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model
model = load_model("cv_model/final.pth", device="cpu")  # ✅ Updated path
classes = ['benign', 'malignant', 'normal']

# Define image transformation (same as training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        predicted = torch.argmax(output, dim=1).item()
        label = classes[predicted]

    # Save image preview
    image_path = "static/uploaded_image.png"
    image.convert("RGB").save(image_path)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": label,
        "image_path": "/" + image_path
    })