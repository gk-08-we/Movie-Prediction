import streamlit as st
import lightgbm as lgb
import numpy as np
import joblib

# Load the trained LightGBM model using the .txt format
model = lgb.Booster(model_file="lightgbm_adjusted_model.txt")

# Load the thresholds for classification
adjusted_thresholds = joblib.load("lightgbm_thresholds.pkl")

# Define the top features used for prediction
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

# Streamlit app title
st.title("Movie Success Prediction App")
st.write("Enter the movie attributes to predict whether it will be a **Hit**, **Flop**, or **Average**.")

# Input fields for each feature
input_values = []
for feature in FEATURES:
    value = st.number_input(f"Enter {feature}:", min_value=0, value=0, step=1)
    input_values.append(value)

# Prediction button
if st.button("Predict"):
    try:
        # Convert input values to a NumPy array and reshape for prediction
        input_array = np.array(input_values).reshape(1, -1)

        # Predict probabilities using the LightGBM model
        probabilities = model.predict(input_array)

        # Apply the adjusted thresholds to determine the class
        if probabilities[0][0] > adjusted_thresholds[0]:
            prediction = "Average"
        elif probabilities[0][1] > adjusted_thresholds[1]:
            prediction = "Flop"
        elif probabilities[0][2] > adjusted_thresholds[2]:
            prediction = "Hit"
        else:
            prediction = ["Average", "Flop", "Hit"][np.argmax(probabilities)]

        # Display the prediction
        st.success(f"The movie is predicted to be: **{prediction.upper()}**")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
