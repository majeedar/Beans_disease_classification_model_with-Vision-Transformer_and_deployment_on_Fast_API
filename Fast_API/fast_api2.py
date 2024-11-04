from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import numpy as np
import io
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Load model and image processor
model_path = "C:/Git_files/bean_leaf_disease/model2/vit-base-beans"
model = ViTForImageClassification.from_pretrained(model_path)
image_processor = ViTImageProcessor.from_pretrained(model_path)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Vision Transformer API! Use /docs to see available endpoints."}

@app.post("/inference")
async def inference(file: UploadFile = File(...)):
    # Ensure uploaded file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    allowed_extensions = {"jpg", "jpeg", "png"}
    if not file.filename.split(".")[-1].lower() in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file extension")

    # Read and process the image
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        inputs = image_processor(images=image, return_tensors="pt")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

    # Retrieve the predicted label and confidence score
    label = model.config.id2label[predicted_class_idx]
    confidence = torch.nn.functional.softmax(logits, dim=-1).max().item()

    logger.info(f"Predicted class: {label}, Confidence: {confidence:.2f}")

    return {
        "predicted_class": label,
        "confidence": confidence
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
