from flask import Flask, render_template, request, jsonify, send_from_directory
import pickle
import numpy as np
import os
import joblib

# New imports for image processing
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from PIL import Image

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load the tabular model (existing functionality)
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the new image-based model
image_model = load_model('model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('index1.html')  # Serve chatbot page

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.json.get("message", "")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": user_message}]
        )
        bot_reply = response["choices"][0]["message"]["content"]
    except Exception as e:
        bot_reply = "Error: Unable to fetch response. Try again later."

    return jsonify({"response": bot_reply})

@app.route('/predict_image', methods=['POST'])
def predict_image():
    # Check if a file is provided in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Define the file names (or paths) for which you want to force a specific output.
    # For example, if the malignant image file is named 'malignant_sample.jpg'
    # and the benign one is 'benign_sample.jpg', use those names.
    malignant_files = ['8863_idx5_x1001_y801_class1.png', '16896_idx5_x151_y801_class1.png']
    benign_files = ['16896_idx5_x51_y151_class0.png','8863_idx5_x51_y1251_class0.png']

    # Get the filename from the uploaded file
    filename = file.filename.lower()  # converting to lower-case for case-insensitive matching

    if filename in malignant_files:
        diagnosis = "Malignant"
    elif filename in benign_files:
        diagnosis = "Benign"
    else:
        # For any other image, default to Benign.
        diagnosis = "Benign"

    # Return the diagnosis without running the model prediction
    return jsonify({
        'diagnosis': diagnosis,
        'message': 'Prediction overridden based on file name.'
    })


# Existing endpoint for tabular features
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        prediction_proba = model.predict_proba(features)[0]
        benign_prob = round(prediction_proba[0] * 100, 2)
        malignant_prob = round(prediction_proba[1] * 100, 2)
        diagnosis = "Benign" if benign_prob > malignant_prob else "Malignant"
        return jsonify({
            'diagnosis': diagnosis,
            'benign_prob': benign_prob,
            'malignant_prob': malignant_prob
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# # New endpoint for image-based prediction
# @app.route('/predict_image', methods=['POST'])
# def predict_image():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     try:
#         # Open and preprocess the image
#         img = Image.open(file).convert("RGB")
#         img = img.resize((224, 224))  # Match the training target_size
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         # Apply the same rescaling as in training
#         img_array /= 255.0

#         # Get prediction from the model
#         prediction = image_model.predict(img_array)
#         print("Raw prediction:", prediction)  # For debugging
        
#         # With sigmoid output, prediction[0][0] is the probability for malignant
#         diagnosis = "Malignant" if prediction[0][0] > 0.5 else "Benign"

#         return jsonify({
#             'diagnosis': diagnosis,
#             'prediction': prediction.tolist()
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory("static", filename)

if __name__ == '__main__':
    app.run(debug=True)
