from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pathlib import Path
import google.generativeai as genai
import pandas as pd
import numpy as np
import cv2
import os
import shutil
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model, load_model

# Load environment and configure Gemini
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Load models and dataset
intensity_model = load_model("weed_intensity_model.keras")


# FastAPI setup
app = FastAPI()

# Replace this with your actual GitHub Pages URL in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://maulikdave27.github.io/weed_removal_frontend_host/"],  # e.g., https://maulikdave.github.io
    allow_methods=["*"],
    allow_headers=["*"],
)

# Image preprocessing
def preprocess_image(image_path: str):
    base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image file")
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = feature_extractor.predict(img)
    return features

# API route
@app.post("/predict")
async def predict_method(
    image: UploadFile = File(...),
    soil_type: str = Form(...),
    humidity: str = Form(...),
    weather: str = Form(...),
    growth_pattern: str = Form(...)
):
    # Save uploaded image
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    image_path = temp_dir / image.filename
    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    # Predict weed intensity
    features = preprocess_image(str(image_path))
    prediction = intensity_model.predict(features)
    intensity = int(np.argmax(prediction) + 1)

    # Gemini prompt (question-based, context-aware)
    prompt = (
        "You are an expert in sustainable weed management. Based on the conditions provided, answer the following:\n\n"
        "1. What type of soil is present? → " + soil_type + "\n"
        "2. What is the weather like in this region? → " + weather + "\n"
        "3. What is the humidity level? → " + humidity + "\n"
        "4. How would you rate the weed intensity on a scale from 1 to 5? → " + str(intensity) + "\n"
        "5. What is the weed growth pattern observed? → " + growth_pattern + "\n\n"
        "Now, considering all the above parameters equally (not prioritizing any one), and choosing from the following methods only:\n"
        "- Biological Weed Control\n"
        "- Mechanical Weed Control\n"
        "- Cultural Weed Control\n"
        "- Chemical Weed Control\n\n"
        "Which **single** weed removal method would you recommend as the most contextually appropriate?\n"
        "**Answer with only the method name**, no justification or explanation."
    )

    # Get Gemini output
    response = gemini_model.generate_content(prompt)
    return {"recommended_method": response.text.strip()}
