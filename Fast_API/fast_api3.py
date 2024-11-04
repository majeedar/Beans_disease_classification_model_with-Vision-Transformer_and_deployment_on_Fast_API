@app.post("/inference")
async def inference(file: UploadFile = File(...)):
    logger.info("Received a file for inference.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    allowed_extensions = {"jpg", "jpeg", "png"}
    if not file.filename.split(".")[-1].lower() in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file extension")

    # Read and process the image
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        logger.info("Image successfully opened and converted to RGB.")

        Resize the image to (224, 224)
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

    logger.info(f"Predicted class: {predicted_class_label}, Confidence: {confidence:.2f}")

    return {
        "predicted_class": predicted_class_label,
        "confidence": confidence
    }
