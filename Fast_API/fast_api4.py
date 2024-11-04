from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import numpy as np
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Load model and image processor
model_path = "C:/Git_files/bean_leaf_disease/model2/vit-base-beans"  # Replace with the correct path to your model directory
model = ViTForImageClassification.from_pretrained(model_path)
image_processor = ViTImageProcessor.from_pretrained(model_path)

# Class mapping for predictions
class_mapping = {
    0: 'angular_leaf_spot',
    1: 'bean_rust',
    2: 'healthy',
    3: 'no_leaf'
}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Vision Transformer API! Use /docs to see available endpoints."}

@app.post("/inference")
async def inference(file: UploadFile = File(...)):
    logger.info("Received a file for inference.")
    
    # Check if the uploaded file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    allowed_extensions = {"jpg", "jpeg", "png"}
    if not file.filename.split(".")[-1].lower() in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file extension")

    # Read and process the image
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        logger.info("Image successfully opened and converted to RGB.")

        # Resize the image to (224, 224)
        image = image.resize((224, 224))
        logger.info("Image resized to 224x224.")

        # Convert image to numpy array and normalize
        image_np = np.array(image) / 255.0  # Normalize to [0, 1]
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        image_np = (image_np - mean) / std
        
        # Convert back to tensor and add batch dimension
        inputs = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0)  # Shape: [1, 3, 224, 224]
        logger.info("Image processed and converted to tensor with shape: {}".format(inputs.shape))
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

    # Perform inference
    try:
        with torch.no_grad():
            outputs = model(inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
    except Exception as e:
        logger.error(f"Error during model inference: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model inference error: {str(e)}")

    # Retrieve the predicted label and confidence score
    predicted_class_label = class_mapping.get(predicted_class_idx, "Unknown Class")
    confidence = torch.nn.functional.softmax(logits, dim=-1).max().item()
    confidence_percentage = confidence * 100  # Convert to percentage

    logger.info(f"Predicted class: {predicted_class_label}, Confidence: {confidence_percentage:.2f}%")

    return {
        "predicted_class": predicted_class_label,
        "confidence": f"{confidence_percentage:.2f}%"  # Format as a percentage
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
