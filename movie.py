import joblib
from flask import Flask, request, jsonify
import numpy as np

# Load the trained model and thresholds
model = joblib.load('final_lightgbm_model.pkl')
adjusted_thresholds = joblib.load('lightgbm_thresholds.pkl')

# Initialize Flask app
app = Flask(__name__)

# Define the top features
FEATURES = [
    'duration', 'director_facebook_likes', 'gross', 'budget',
    'cast_total_facebook_likes', 'actor_1_facebook_likes',
    'actor_3_facebook_likes', 'actor_2_facebook_likes',
    'facenumber_in_poster', 'content_rating_R'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        data = request.get_json()

        # Extract features from the input JSON
        input_features = [float(data[feature]) for feature in FEATURES]

        # Ensure all input values are positive whole numbers
        if any(value < 0 or not float(value).is_integer() for value in input_features):
            return jsonify({"error": "All input values must be positive whole numbers."}), 400

        # Convert input features to a 2D array
        input_array = np.array(input_features).reshape(1, -1)

        # Get prediction probabilities
        probabilities = model.predict_proba(input_array)[0]

        # Apply adjusted thresholds
        if probabilities[0] > adjusted_thresholds[0]:
            prediction = 'Average'
        elif probabilities[1] > adjusted_thresholds[1]:
            prediction = 'Flop'
        elif probabilities[2] > adjusted_thresholds[2]:
            prediction = 'Hit'
        else:
            prediction = ['Average', 'Flop', 'Hit'][np.argmax(probabilities)]

        # Return the prediction in a user-friendly text format
        return jsonify({"message": f"The movie is predicted to be {prediction.upper()}."})

    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
