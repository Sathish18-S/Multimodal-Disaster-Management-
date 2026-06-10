from flask import Flask, request, jsonify
import json
import math
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This handles CORS for your React app

# ---------- Configuration ----------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_tweets.json")

# ---------- Load fabricated tweets with error handling ----------
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

# ---------- Helper to calculate distance ----------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ---------- Find nearest region ----------
def find_nearest_region(lat, lon):
    nearest_city, min_dist = None, float("inf")
    for city, (clat, clon) in REGIONS.items():
        dist = haversine(lat, lon, clat, clon)
        if dist < min_dist:
            min_dist = dist
            nearest_city = city
    return nearest_city, min_dist

# ---------- Input validation ----------
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

# ---------- Routes ----------
@app.route("/")
def index():
    return jsonify({
        "message": "Disaster Intelligence API is running",
        "available_regions": list(REGIONS.keys()),
        "status": "OK"
    })

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

# Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "regions_loaded": len(REGIONS),
        "tweets_loaded": sum(len(tweets) for tweets in TWEETS.values()) if TWEETS else 0
    })

# Get available regions
@app.route("/regions", methods=["GET"])
def get_regions():
    return jsonify({
        "regions": REGIONS,
        "total_regions": len(REGIONS)
    })

if __name__ == "__main__":
    # Better configuration for production
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))
    
    print(f"🚀 Disaster Intelligence API starting...")
    print(f"📊 Loaded regions: {list(REGIONS.keys())}")
    print(f"🐦 Total tweets loaded: {sum(len(tweets) for tweets in TWEETS.values()) if TWEETS else 0}")
    
    app.run(
        host='0.0.0.0',  # Allow connections from any IP
        port=port,
        debug=debug_mode
    )