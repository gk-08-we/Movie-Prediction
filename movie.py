import streamlit as st
import joblib
import numpy as np

# Load the trained LightGBM model and thresholds
model = joblib.load("lightgbm_adjusted_model.pkl")
adjusted_thresholds = joblib.load("lightgbm_thresholds.pkl")

# Define the top features
FEATURES = [
    "duration", 
    "director_facebook_likes", 
    "gross", 
    "budget",
    "cast_total_facebook_likes", 
    "actor_1_facebook_likes",
    "actor_3_facebook_likes", 
    "actor_2_facebook_likes",
    "facenumber_in_poster", 
    "content_rating_R"
]

# Streamlit app
st.title("Movie Success Prediction App")

st.write("Enter the feature values to predict if a movie will be a Hit, Flop, or Average.")

# Input fields for the features
input_values = []
for feature in FEATURES:
    value = st.number_input(f"Enter {feature}:", min_value=0, value=0, step=1)
    input_values.append(value)

# Prediction button
if st.button("Predict"):
    try:
        # Convert inputs to NumPy array and reshape for prediction
        input_array = np.array(input_values).reshape(1, -1)

        # Predict probabilities
        probabilities = model.predict_proba(input_array)[0]

        # Apply adjusted thresholds
        if probabilities[0] > adjusted_thresholds[0]:
            prediction = "Average"
        elif probabilities[1] > adjusted_thresholds[1]:
            prediction = "Flop"
        elif probabilities[2] > adjusted_thresholds[2]:
            prediction = "Hit"
        else:
            prediction = ["Average", "Flop", "Hit"][np.argmax(probabilities)]

        # Display the prediction
        st.success(f"The movie is predicted to be: **{prediction.upper()}**")
    except Exception as e:
        st.error(f"Error: {str(e)}")
