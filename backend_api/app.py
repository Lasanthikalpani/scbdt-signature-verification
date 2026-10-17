from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import Swagger  # <--- Swagger UI සඳහා
import cv2
import numpy as np
import tensorflow as tf
import os
import sys

# src ෆෝල්ඩරයේ ඇති කේත import කිරීමට
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.feature_extraction import FeatureExtractor
from src.rag_engine import RAGEngine

app = Flask(__name__)
CORS(app)
Swagger(app) # <--- Swagger එකතු කිරීම

# ආකෘති සහ එන්ජින් Load කිරීම
model = tf.keras.models.load_model('models/HDSV-Net.h5')
extractor = FeatureExtractor()
rag_engine = RAGEngine()

@app.route('/api/verify', methods=['POST'])
def verify_signature():
    """
    Signature Forgery Detection API (SCBDT Theory)
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: image
        in: formData
        type: file
        required: true
        description: The signature image file to verify (PNG, JPG)
    responses:
      200:
        description: A JSON object containing the verification result
        examples:
          application/json: {
            "is_genuine": true,
            "cbi_total": 0.206,
            "cbi_scores": {"CBI_1": 0.065, "CBI_6": 0.134},
            "explanation": "The signature matches the reference.",
            "status": "GENUINE"
          }
      400:
        description: Invalid input provided
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return jsonify({"error": "Invalid image format"}), 400

    # 1. ගණිතමය ලක්ෂණ (CBI1-CBI5)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)
    
    features = {
        "CBI_1": extractor.get_slant(binary) / 90.0,
        "CBI_2": min(extractor.get_size_width(binary)[0] / 50000.0, 1.0),
        "CBI_3": min(extractor.get_mam(img, binary) / 50.0, 1.0),
        "CBI_4": extractor.get_binary_features(binary),
        "CBI_5": np.mean(np.abs(extractor.get_moments(binary)))
    }
    
    # 2. HDSV-Net (CBI6)
    img_resized = cv2.resize(binary, (64, 64)) / 255.0
    img_input = img_resized.reshape(1, 64, 64, 1)
    cbi_6 = float(model.predict(img_input, verbose=0)[0][0])
    
    # 3. CBI_total ගණනය කිරීම
    weights = {"CBI_1": 0.20, "CBI_2": 0.15, "CBI_3": 0.15, "CBI_4": 0.15, "CBI_5": 0.15, "CBI_6": 0.20}
    features["CBI_6"] = 1.0 - cbi_6  # CBI6 ප්‍රතිලෝම කිරීම (1 = Genuine, 0 = Forged)
    cbi_total = sum(weights[k] * features[k] for k in weights)
    
    # 4. තීරණය සහ RAG පැහැදිලි කිරීම
    is_genuine = cbi_total < 0.25
    explanation = rag_engine.generate_explanation(features, cbi_6)
    
    return jsonify({
        "is_genuine": is_genuine,
        "cbi_total": round(cbi_total, 3),
        "cbi_scores": {k: round(v, 3) for k, v in features.items()},
        "explanation": explanation,
        "status": "GENUINE" if is_genuine else "FORGED"
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)