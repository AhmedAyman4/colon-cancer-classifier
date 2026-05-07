import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import os

import keras
from keras.layers import Dense, BatchNormalization

# Fix for Keras 3 "quantization_config" deserialization issue
# This happens when a model saved with a newer Keras version is loaded in an environment
# where the layers do not yet expect the 'quantization_config' argument.
original_dense_init = Dense.__init__
def patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    return original_dense_init(self, *args, **kwargs)
Dense.__init__ = patched_dense_init

original_bn_init = BatchNormalization.__init__
def patched_bn_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    return original_bn_init(self, *args, **kwargs)
BatchNormalization.__init__ = patched_bn_init

app = FastAPI(title="Colon Cancer Classifier API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = None

@app.on_event("startup")
def load_model():
    global model
    model_path = os.path.join(os.path.dirname(__file__), "model/model.keras")
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found at {model_path}")
    
    print("Loading model with compatibility patches...")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully")

# Class names mapping from training
CLASS_NAMES = ["colon_aca", "colon_n"]
LABEL_MAPPING = {
    "colon_aca": "Colon Adenocarcinoma",
    "colon_n": "Colon Benign Tissue"
}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Resize to 224x224 as per training parameters
        image = image.resize((224, 224))
        
        # Convert to array and add batch dimension
        img_array = np.array(image).astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply the same preprocessing as in training (ResNet50 style)
        img_array = preprocess_input(img_array)
        
        # Predict
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        tech_name = CLASS_NAMES[predicted_class_idx]
        display_name = LABEL_MAPPING[tech_name]
        
        # Format results
        results = {
            "prediction": tech_name,
            "display_name": display_name,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                LABEL_MAPPING[CLASS_NAMES[i]]: round(float(predictions[0][i]) * 100, 2)
                for i in range(len(CLASS_NAMES))
            }
        }
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files (this should be last)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Port 7860 is the default for Hugging Face Spaces
    uvicorn.run(app, host="0.0.0.0", port=7860)
