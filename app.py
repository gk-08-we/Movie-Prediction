import numpy as np
import pickle
import pandas as pd
import streamlit as st

st.title('Movie Success Predictor')
st.write("Classify movies as 'Hit', 'Average', or 'Flop' based on their IMDB scores.")

# Load pre-trained model
with open('movie_model.pkl', 'rb') as file:
    model = pickle.load(file)

def predict(imdb_score):
    # Categorize IMDB Scores into 'Hit', 'Average', or 'Flop'
    if imdb_score <= 3:
        return 'Flop'
    elif 3 < imdb_score <= 6:
        return 'Average'
    else:
        return 'Hit'

def main():
    option = st.selectbox("Choose an input method:", ("Manual Input", "Upload CSV"))

    if option == "Manual Input":
        imdb_score = st.number_input("Enter IMDB Score (1 to 10):", min_value=1.0, max_value=10.0, step=0.1)
        if st.button("Predict"):
            prediction = predict(imdb_score)
            st.success(f"The movie is predicted to be: **{prediction}**")

    elif option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV file (must contain 'imdb_score' column):", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'imdb_score' in df.columns:
                    st.write("Dataset Preview:")
                    st.dataframe(df.head())

                    # Apply classification
                    df['Prediction'] = df['imdb_score'].apply(predict)

                    st.write("Predicted Results:")
                    st.dataframe(df[['imdb_score', 'Prediction']])

                    # Option to download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name="movie_predictions.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("The uploaded file does not contain the 'imdb_score' column.")
            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")

if __name__ == '__main__':
    main()
