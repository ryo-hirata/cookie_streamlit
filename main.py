from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
from io import BytesIO
from keras.models import load_model
from keras import backend as K
import json
from fastapi.encoders import jsonable_encoder
import base64
import matplotlib.pyplot as plt


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def tensor_to_list(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor.tolist()
    return tensor

def r_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])

app = FastAPI()

z_dim = 50  # Use the correct z_dim value
# Load the Keras model
model = load_model('model.h5', custom_objects={'r_loss': r_loss})

# Anomaly score threshold (you can adjust this value)
anomaly_score_threshold = 0.027

def process_image(file: UploadFile):
    contents = file.file.read()
    pil_image = Image.open(BytesIO(contents)).convert("RGB")
    pil_image = pil_image.resize((400, 300))
    numpy_image = np.array(pil_image)

    # Get the actual size of the image
    image_size = numpy_image.shape
    
    tensor_image = numpy_image.astype(np.float32) / 255.0
    tensor_image = np.expand_dims(tensor_image, axis=0)
    
    return tensor_image, image_size  # Return the NumPy array and its size

def visualize_images(original, reconstructed, difference):
    # Convert NumPy arrays to PIL Images
    original_pil = Image.fromarray((original[0] * 255).astype(np.uint8))
    reconstructed_pil = Image.fromarray((reconstructed[0] * 255).astype(np.uint8))
    difference_pil = Image.fromarray((difference[0] * 255).astype(np.uint8))

    # Save the PIL Images to BytesIO objects
    original_io = BytesIO()
    reconstructed_io = BytesIO()
    difference_io = BytesIO()
    original_pil.save(original_io, format='PNG')
    reconstructed_pil.save(reconstructed_io, format='PNG')
    difference_pil.save(difference_io, format='PNG')

    # Convert BytesIO objects to base64-encoded strings
    original_base64 = base64.b64encode(original_io.getvalue()).decode('utf-8')
    reconstructed_base64 = base64.b64encode(reconstructed_io.getvalue()).decode('utf-8')
    difference_base64 = base64.b64encode(difference_io.getvalue()).decode('utf-8')

    # Return the base64-encoded strings
    return original_base64, reconstructed_base64, difference_base64

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Process the uploaded image
    input_image_tensor, image_size = process_image(file)

    # Inference using the model
    reconstructed_image = model.predict(input_image_tensor)

    # Calculate anomaly score
    diff_images = np.absolute(reconstructed_image - input_image_tensor)
    anomaly_score = np.mean(np.abs(diff_images))

    # Visualize the original image, reconstructed image, and their difference
    original_base64, reconstructed_base64, difference_base64 = visualize_images(input_image_tensor, reconstructed_image, diff_images)

    # Classify as defective or non-defective based on the threshold
    is_defective = anomaly_score > anomaly_score_threshold

    # Convert the NumPy array to a list before serializing to JSON
    reconstructed_image_list = tensor_to_list(reconstructed_image)

    # Convert the tensor in anomaly_score to Python float
    anomaly_score = anomaly_score.item()

    # JSON response content
    content = {
        "result": "success",
        "is_defective": bool(is_defective),  # Convert to standard bool
        "reconstructed_image": reconstructed_image_list,
        "anomaly_score": anomaly_score,
        "input_image_data": tensor_to_list(input_image_tensor),  # Convert NumPy array to list
        "visualizations": {
            "original": original_base64,
            "reconstructed": reconstructed_base64,
            "difference": difference_base64
        }
    }

    # Return JSON response
    return JSONResponse(content=content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
