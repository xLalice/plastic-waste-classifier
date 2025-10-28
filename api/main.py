from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os
from dotenv import load_dotenv
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI(
    title="Waste Classifier API",
    description="An API to classify common types of waste using a Deep Learning model.",
    version="1.0.0"
)

load_dotenv()

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

origins = [
    FRONTEND_URL,
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

CLASS_NAMES = ['Aluminum_Cans', 'Cardboard', 'Glass_Bottles', 'HDPE_Containers', 'PET_Bottles']


SCRIPT_DIR = Path(__file__).parent.resolve()

MODEL_PATH = SCRIPT_DIR / 'plastic_classifier_model_FINETUNED.keras'
model = None

print("--- API Starting Up ---")
if not os.path.exists(MODEL_PATH):
    print(f"FATAL ERROR: Model file not found at path: '{MODEL_PATH}'")
    print("Please make sure you have downloaded the .keras file from Colab and placed it here.")
else:
    print(f"Model file found at '{MODEL_PATH}'. Loading full model...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("SUCCESS: Full model loaded successfully!")
    except Exception as e:
        print(f"FATAL ERROR: Failed to load model. Error: {e}")
        model = None
print("-----------------------")
# --- THIS IS THE END OF THE FIX ---


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    This function is now 100% identical to the training pipeline.
    It prepares the image with [0, 255] pixels as a float32 array,
    and the loaded .keras model will handle the [-1, 1] scaling internally.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # --- START OF PREPROCESSING FIX ---
    # 1. Resize using the *exact same* interpolation as the training data
    image = image.resize((224, 224), resample=Image.NEAREST)
    
    # 2. Convert to array using Keras's utility to get the correct
    #    data type (float32) and channel order.
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    
    # 3. Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # 4. **THE CRITICAL FIX**: Apply the same [-1, 1] scaling
    #    that the ImageDataGenerator used during training.
    processed_image = preprocess_input(image_array)
    # --- END OF PREPROCESSING FIX ---
    
    return processed_image # This now returns [-1, 1] data as float32

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model is not loaded. Check server logs.")

    image_bytes = await file.read()

    try:
        Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file provided.")

    try:
        # 1. Preprocess image (converts to [1, 224, 224, 3] array with [-1, 1] pixels, float32)
        processed_image = preprocess_image(image_bytes)
        
        # 2. Predict
        # The model *itself* will now apply the [-1, 1] scaling
        # before the MobileNetV2 base model runs.
        prediction = model.predict(processed_image)
        
        # 3. Get results
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = float(np.max(prediction))

        # --- THIS IS A SECONDARY FIX ---
        # Return confidence as a number (e.g., 90.95), not a string.
        # This is much better for your React frontend.
        return {
            "prediction": predicted_class_name,
            "confidence": confidence * 100 
        }
        # --- END OF SECONDARY FIX ---
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "Waste Classifier API is running."}
