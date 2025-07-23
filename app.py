from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from model import EmotionPredictor, validate_model_file
import os
from typing import Dict, Any, Optional, Union

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = 'models/resnet50_emotion_webapp.pth'
DEVICE = 'cpu'  # Use 'cuda' if GPU is available

# Global model instance
predictor: Optional[EmotionPredictor] = None

def initialize_model() -> bool:
    """Initialize the emotion prediction model with minimal preprocessing"""
    global predictor
    
    try:
        print("🚀 Initializing Emotion Recognition API with Minimal Preprocessing...")
        
        if not validate_model_file(MODEL_PATH):
            return False
        
        predictor = EmotionPredictor(MODEL_PATH, DEVICE)
        print("🎭 Emotion Recognition API with Minimal Preprocessing initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        return False

@app.route('/api/health', methods=['GET'])
def health_check() -> Response:
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'preprocessing': 'Minimal Quality Preserving',
        'model_info': predictor.get_model_info() if predictor else None
    })

@app.route('/api/predict', methods=['POST'])
def predict_emotion() -> Union[Response, tuple[Response, int]]:
    """Main prediction endpoint with minimal preprocessing"""
    try:
        print("\n" + "="*60)
        print("🔮 NEW PREDICTION REQUEST WITH MINIMAL PREPROCESSING")
        print("="*60)
        
        if predictor is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        print(f"📁 Received file: {file.filename}")
        print(f"📊 File content type: {file.content_type}")
        
        # Make prediction with minimal preprocessing
        result = predictor.predict_emotion(file)
        
        print(f"🎯 Final prediction: {result['emotion']}")
        print(f"💯 Confidence: {result['confidence']}%")
        print(f"🔄 Preprocessing: {result['preprocessing_type']}")
        print("="*60)
        
        # Ensure all values are properly typed for JSON serialization
        formatted_result = {
            'emotion': str(result['emotion']),
            'confidence': float(result['confidence']),
            'color': str(result['color']),
            'preprocessing_applied': bool(result['preprocessing_applied']),
            'preprocessing_type': str(result['preprocessing_type']),
            'all_probabilities': [
                {
                    'emotion': str(prob['emotion']),
                    'probability': float(prob['probability']),
                    'color': str(prob['color'])
                }
                for prob in result['all_probabilities']
            ]
        }
        
        return jsonify(formatted_result)
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def get_model_info() -> Union[Response, tuple[Response, int]]:
    """Get model information"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify(predictor.get_model_info())

if __name__ == '__main__':
    if initialize_model():
        print("\n🌐 Starting Flask Server with Minimal Preprocessing...")
        print("🔗 Backend API: http://localhost:5000")
        print("🔗 Health check: http://localhost:5000/api/health")
        print("🔗 Features: Minimal Quality Preserving Preprocessing")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to start server. Please check the model file.")
