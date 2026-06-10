import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import cv2
from PIL import Image
import io
import base64
import json
import math

app = Flask(__name__)
CORS(app)  # This handles CORS for your React app

# ---------- Configuration ----------
# Disaster Intelligence Configuration
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_tweets.json")

# Image Classification Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = "models/multimodalmodel.keras"
LABEL_NAMES = ["No damage", "Mild", "Severe", "Help needed"]

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- Initialize models ----------
model = None
text_vectorizer = None

# ---------- Define region coordinates ----------
REGIONS = {
    "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558),
    "Madurai": (9.9252, 78.1198),
    "Tirunelveli": (8.7139, 77.7567),
    "Salem": (11.6643, 78.1460),
    "Mumbai": (19.0760, 72.8777),
    "Kolkata": (22.5726, 88.3639)
}

# ---------- Load fabricated tweets ----------
def load_tweets():
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in data file {DATA_PATH}")
        return {}
    except Exception as e:
        print(f"Error loading tweets: {e}")
        return {}

TWEETS = load_tweets()

# ---------- Load ML models ----------
def load_models():
    """Load the trained model and text vectorizer"""
    global model, text_vectorizer
    
    try:
        if os.path.exists(MODEL_PATH):
            print("Loading multimodal model...")
            model = load_model(MODEL_PATH)
            print("Model loaded successfully!")
        else:
            print(f"Warning: Model not found at {MODEL_PATH}")
            print("Using mock predictions for demonstration.")
            
        # Initialize text vectorizer
        max_tokens = 10000
        max_len = 128
        
        text_vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode='int',
            output_sequence_length=max_len
        )
        
        # For demo purposes, create a simple vocabulary
        demo_texts = [
            "flood water damage severe help needed rescue",
            "mild flooding roads wet",
            "no damage clear normal",
            "water accumulation minor flooding"
        ]
        text_vectorizer.adapt(demo_texts)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using mock predictions.")

# ---------- Helper functions for disaster intelligence ----------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def find_nearest_region(lat, lon):
    nearest_city, min_dist = None, float("inf")
    for city, (clat, clon) in REGIONS.items():
        dist = haversine(lat, lon, clat, clon)
        if dist < min_dist:
            min_dist = dist
            nearest_city = city
    return nearest_city, min_dist

def validate_coordinates(lat, lon):
    if lat is None or lon is None:
        return False, "Latitude and longitude are required"
    
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return False, "Invalid latitude or longitude format"
    
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return False, "Coordinates out of valid range"
    
    return True, (lat, lon)

# ---------- Helper functions for image classification ----------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_text(text):
    """Preprocess text for model prediction"""
    text = str(text).lower().strip()
    # Vectorize the text
    vectorized = text_vectorizer([text])
    return vectorized.numpy()

def mock_prediction(text):
    """Generate mock predictions when model is not available"""
    text_lower = text.lower()
    
    # Simple rule-based mock predictions
    if any(word in text_lower for word in ['help', 'rescue', 'emergency', 'stranded', 'trapped']):
        scores = [0.1, 0.1, 0.2, 0.6]  # High for "Help needed"
    elif any(word in text_lower for word in ['severe', 'widespread', 'submerged', 'damage', 'destroyed']):
        scores = [0.1, 0.2, 0.6, 0.1]  # High for "Severe"
    elif any(word in text_lower for word in ['mild', 'minor', 'accumulation', 'localized', 'caution']):
        scores = [0.2, 0.6, 0.1, 0.1]  # High for "Mild"
    else:
        scores = [0.7, 0.2, 0.05, 0.05]  # High for "No damage"
    
    # Add some randomness
    scores = np.array(scores) + np.random.normal(0, 0.05, 4)
    scores = np.clip(scores, 0, 1)
    scores = scores / np.sum(scores)  # Normalize
    
    return scores

# ---------- Routes ----------
@app.route("/")
def index():
    return jsonify({
        "message": "Disaster Intelligence & Damage Assessment API is running",
        "available_regions": list(REGIONS.keys()),
        "ml_model_loaded": model is not None,
        "status": "OK"
    })

# ---------- Disaster Intelligence Routes ----------
@app.route("/get_tweets", methods=["POST"])
def get_tweets():
    try:
        data = request.get_json()
        
        # Check if request has JSON data
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        region = data.get("region")
        lat = data.get("lat")
        lon = data.get("lon")

        # Case 1: Region name provided
        if region:
            if region not in REGIONS:
                return jsonify({
                    "error": f"Region '{region}' not found. Available regions: {list(REGIONS.keys())}"
                }), 404
            
            tweets = TWEETS.get(region, [])
            return jsonify({
                "region": region, 
                "distance_km": None, 
                "tweets": tweets,
                "total_tweets": len(tweets)
            })

        # Case 2: Coordinates provided
        if lat is not None and lon is not None:
            # Validate coordinates
            is_valid, validation_result = validate_coordinates(lat, lon)
            if not is_valid:
                return jsonify({"error": validation_result}), 400
            
            lat, lon = validation_result
            nearest_city, min_dist = find_nearest_region(lat, lon)
            tweets = TWEETS.get(nearest_city, [])
            
            return jsonify({
                "region": nearest_city,
                "distance_km": round(min_dist, 1),
                "tweets": tweets,
                "total_tweets": len(tweets),
                "searched_coordinates": {"lat": lat, "lon": lon}
            })

        # No valid input provided
        return jsonify({
            "error": "Please provide either 'region' name or 'lat' and 'lon' coordinates",
            "available_regions": list(REGIONS.keys())
        }), 400

    except Exception as e:
        print("Error in /get_tweets:", str(e))
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

# ---------- Damage Assessment Routes ----------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if files are present
        if 'image' not in request.files or 'text' not in request.form:
            return jsonify({'error': 'Missing image or text input'}), 400
        
        image_file = request.files['image']
        text_input = request.form['text']
        
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if not text_input.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        if image_file and allowed_file(image_file.filename):
            # Save uploaded image
            filename = secure_filename(image_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(image_path)
            
            # Preprocess inputs
            image_array = preprocess_image(image_path)
            text_array = preprocess_text(text_input)
            
            # Make prediction
            if model is not None:
                # Real prediction
                prediction = model.predict([image_array, text_array])
                confidence_scores = prediction[0]
                is_mock = False
            else:
                # Mock prediction based on text
                confidence_scores = mock_prediction(text_input)
                is_mock = True
            
            # Get top prediction
            predicted_class_idx = np.argmax(confidence_scores)
            predicted_class = LABEL_NAMES[predicted_class_idx]
            confidence = float(confidence_scores[predicted_class_idx])
            
            # Prepare results
            results = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_scores': {
                    LABEL_NAMES[i]: float(score) for i, score in enumerate(confidence_scores)
                },
                'is_mock': is_mock
            }
            
            # Convert image to base64 for display
            with open(image_path, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            results['image_preview'] = f"data:image/jpeg;base64,{image_base64}"
            results['input_text'] = text_input
            
            return jsonify(results)
        
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ---------- Common Routes ----------
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "regions_loaded": len(REGIONS),
        "tweets_loaded": sum(len(tweets) for tweets in TWEETS.values()) if TWEETS else 0,
        "ml_model_loaded": model is not None,
        "using_mock_predictions": model is None
    })

@app.route("/regions", methods=["GET"])
def get_regions():
    return jsonify({
        "regions": REGIONS,
        "total_regions": len(REGIONS)
    })

# Initialize the application
if __name__ == "__main__":
    # Load both datasets and models
    print(f"🚀 Disaster Intelligence & Damage Assessment API starting...")
    print(f"📊 Loaded regions: {list(REGIONS.keys())}")
    print(f"🐦 Total tweets loaded: {sum(len(tweets) for tweets in TWEETS.values()) if TWEETS else 0}")
    
    # Load ML models
    load_models()
    
    # Better configuration for production
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))
    
    app.run(
        host='0.0.0.0',  # Allow connections from any IP
        port=port,
        debug=debug_mode
    )