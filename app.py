import pandas as pd

try:
    import streamlit as st
except ModuleNotFoundError:
    st.write("Streamlit is not installed in the current environment.")
    raise ModuleNotFoundError("Streamlit is not installed. Please install it using 'pip install streamlit'.")

from sklearn.preprocessing import LabelEncoder

# Debugging output
st.write("Initializing Streamlit application...")

# Streamlit App
st.title("Movie Success Predictor")
st.write("Classify movies as 'Hit', 'Average', or 'Flop' based on their IMDB scores.")

# Input Options
option = st.selectbox("Choose an input method:", ("Manual Input", "Upload CSV"))
st.write(f"Selected option: {option}")

# Categorize IMDB Scores into 'Hit', 'Average', or 'Flop'
def classify_movie(score):
    st.write(f"Classifying score: {score}")
    if score <= 3:
        return 'Flop'
    elif 3 < score <= 6:
        return 'Average'
    else:
        return 'Hit'

if option == "Manual Input":
    imdb_score = st.number_input("Enter IMDB Score:", min_value=0.0, max_value=10.0, step=0.1)
    st.write(f"IMDB Score entered: {imdb_score}")
    if st.button("Predict"):
        category = classify_movie(imdb_score)
        st.write(f"The movie is predicted to be: **{category}**")

elif option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file:", type="csv")
    if uploaded_file is not None:
        st.write("File uploaded successfully.")
        try:
            df = pd.read_csv(uploaded_file)
            st.write("CSV file loaded into DataFrame.")

            # Check for 'imdb_score' column
            if 'imdb_score' in df.columns:
                st.write("Dataset Preview:")
                st.dataframe(df.head())

                # Apply classification
                df['Classify'] = df['imdb_score'].apply(classify_movie)

                st.write("Predicted Results:")
                st.dataframe(df[['imdb_score', 'Classify']])

                # Option to download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Results",
                    data=csv,
                    file_name="movie_predictions.csv",
                    mime="text/csv",
                )
            else:
                st.error("The uploaded CSV does not contain the 'imdb_score' column.")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
            st.write("Traceback:", e)
