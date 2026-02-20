import streamlit as st
from src.pipline.prediction_pipeline import SpotifyData,PredictionPipeline
st.title("Spotify Tracks Classification")
st.write("This is a simple Streamlit app for Spotify Tracks Classification.")
import time

dancebility = st.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
key = st.number_input("Key", min_value=0, max_value=11, value=5)
loudness = st.number_input("Loudness", min_value=-60.0, max_value=0.0, value=-10.0)
mode = st.selectbox("Mode", [0, 1], index=1)
speechiness = st.slider("Speechiness", 0.0, 1.0, 0.5)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.5)
liveness = st.slider("Liveness", 0.0, 1.0, 0.5)
valence = st.slider("Valence", 0.0, 1.0, 0.5)
tempo = st.number_input("Tempo", min_value=0.0, max_value=250.0, value=120.0)
duration_ms = st.number_input("Duration (ms)", min_value=0, max_value=600000, value=210000)
time_signature = st.number_input("Time Signature", min_value=0, max_value=7, value=4)
chorus_hit = st.number_input("Chorus Hit", min_value=0.0, max_value=300.0, value=60.0)
sections = st.number_input("Sections", min_value=1, max_value=50, value=8)
test_song = SpotifyData(
    danceability=dancebility,
    energy=energy,
    key=key,
    loudness=loudness,
    mode=mode,
    speechiness=speechiness,
    acousticness=acousticness,
    instrumentalness=instrumentalness,
    liveness=liveness,
    valence=valence,
    tempo=tempo,
    duration_ms=duration_ms,
    time_signature=time_signature,
    chorus_hit=chorus_hit,
    sections=sections
)
if st.button("Predict"):
    prediction_pipeline=PredictionPipeline()
    start_time = time.time()
    status=prediction_pipeline.predict(test_song)
    end_time = time.time()
    st.write(f"Prediction completed in {end_time - start_time:.2f} seconds.")
    if status==1:
        st.success("The track is likely to be a hit!")
    else:
        st.warning("The track is less likely to be a hit.")
