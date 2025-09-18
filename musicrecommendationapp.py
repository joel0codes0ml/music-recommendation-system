import streamlit as st
import numpy as np
import pickle

# Load model & scaler
model = pickle.load(open("music_genre_model.sav", "rb"))
scaler = pickle.load(open("scaler.sav", "rb"))

st.title("🎶 Music Genre Classifier")

st.write("Enter the audio features of a song to predict its genre.")

# Inputs
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
loudness = st.slider("Loudness (dB)", -60.0, 0.0, -5.0)
speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.3)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
valence = st.slider("Valence", 0.0, 1.0, 0.5)
tempo = st.slider("Tempo (BPM)", 50.0, 200.0, 120.0)

# Predict
if st.button("Predict Genre"):
    features = np.array([[danceability, energy, loudness, speechiness,
                          acousticness, instrumentalness, valence, tempo]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    st.success(f"🎵 Predicted Genre: **{prediction[0]}**")

