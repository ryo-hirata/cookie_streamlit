from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
from io import BytesIO
from keras.models import load_model
from keras import backend as K
import json

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

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Process the uploaded image
    input_image_tensor, image_size = process_image(file)

    # Inference using the model
    reconstructed_image = model.predict(input_image_tensor)

    # Convert the NumPy array to a PIL Image for easy visualization
    reconstructed_image_pil = Image.fromarray((reconstructed_image[0] * 255).astype(np.uint8).transpose(1, 2, 0))

    # Ensure the shapes are compatible
    input_image_tensor = input_image_tensor.transpose(0, 3, 1, 2)

    # Calculate anomaly score
    mse = ((input_image_tensor - reconstructed_image) ** 2).mean()
    anomaly_score = mse  # Adjust as needed

    # Classify as defective or non-defective based on the threshold
    is_defective = anomaly_score > anomaly_score_threshold

    # Convert the NumPy array to a list before serializing to JSON
    reconstructed_image_list = tensor_to_list(reconstructed_image)
    
    # Convert the tensor in anomaly_score to Python float
    anomaly_score = anomaly_score.item()

    # Use CustomJSONEncoder
    content = {
        "result": "success",
        "is_defective": is_defective,
        "reconstructed_image": reconstructed_image_list,
        "anomaly_score": anomaly_score,
        "input_image_data": tensor_to_list(input_image_tensor)  # Convert NumPy array to list
    }

    # FastAPI will automatically convert content to JSON
    return JSONResponse(content=content)
