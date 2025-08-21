import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model and expected feature names
try:
    with open("cricket_model.pkl", "rb") as f:
        model, expected_features = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'cricket_model.pkl' not found. Please run the model training script first.")
    st.stop()

st.set_page_config(layout="centered", page_title="Cricket Player Run Predictor 🏏")

st.title("🏏 Cricket Player Run Predictor")
st.markdown("Predict a player's runs scored based on their career statistics and match context.")

# --- Collect User Inputs ---
st.header("Player Statistics")

col1, col2 = st.columns(2)

with col1:
    value = st.slider("Player's IPL Value (in Cr)", 0.0, 20.0, 1.0, step=0.1)
    matches = st.number_input("Total Matches Played", min_value=0, max_value=300, value=50)
    innings = st.number_input("Innings Batted", min_value=0, max_value=300, value=40)
    not_outs = st.number_input("Not Outs", min_value=0, max_value=100, value=5)
    hundreds = st.number_input("100s Scored", min_value=0, max_value=20, value=1)
    fifties = st.number_input("50s Scored", min_value=0, max_value=50, value=5)
    fours = st.number_input("4s Hit", min_value=0, max_value=500, value=60)

with col2:
    sixes = st.number_input("6s Hit", min_value=0, max_value=500, value=30)
    bat_avg = st.number_input("Batting Average", min_value=0.0, max_value=100.0, value=35.0, step=0.1)
    bat_sr = st.number_input("Batting Strike Rate", min_value=0.0, max_value=300.0, value=120.0, step=0.1)
    ducks = st.number_input("Ducks", min_value=0, max_value=100, value=2)
    age = st.number_input("Age (Years)", min_value=15, max_value=45, value=28)
    highest_inn_score = st.number_input("Highest Innings Score", min_value=0, max_value=200, value=50)
    
st.header("Player Type & Background")

col3, col4, col5 = st.columns(3)

with col3:
    team = st.selectbox("Team", ['CSK', 'DC', 'GT', 'KKR', 'LSG', 'MI', 'PBKS', 'RCB', 'RR', 'SRH', 'Other'], index=0)
with col4:
    player_type = st.selectbox("Player Type", ['Batsman', 'All-Rounder', 'Bowler', 'Wicketkeeper', 'Other'], index=0)
with col5:
    batting_style = st.selectbox("Batting Style", ['Right Handed', 'Left Handed', 'Other'], index=0)

col6, col7 = st.columns(2)
with col6:
    national_side = st.selectbox("National Side", ['India', 'England', 'Australia', 'South Africa', 'New Zealand', 'West Indies', 'Afghanistan', 'Bangladesh', 'Sri Lanka', 'Other'], index=0)
with col7:
    bowling_style = st.selectbox("Bowling Style", [
        'Leg break', 'Right-arm medium-fast', 'Right-arm fast-medium',
        'Left-arm orthodox', 'Off break', 'Right-arm offbreak',
        'Slow left-arm chinaman', 'Right-arm fast', 'Left-arm fast-medium',
        'Legbreak googly', 'Left-arm medium-fast', 'Right-arm medium',
        'Left-arm medium', 'Other', 'No Bowling' # Added 'No Bowling' for players who don't bowl
    ], index=13) # 'Other' as default

# --- Prepare Input for Prediction ---
# Create a dictionary for numerical inputs
user_input_numerical = {
    'ValueinCR': value,
    'MatchPlayed': matches,
    'InningsBatted': innings,
    'NotOuts': not_outs,
    '100s': hundreds,
    '50s': fifties,
    '4s': fours,
    '6s': sixes,
    'BattingAVG': bat_avg,
    'BattingS/R': bat_sr,
    'Ducks': ducks,
    'Age': age,
    'HighestInnScore': highest_inn_score
}

# Initialize a dictionary to hold all feature values, setting all to 0 initially
# This ensures all one-hot encoded columns are present, even if not selected by the user.
model_input_dict = {feature: 0 for feature in expected_features}

# Populate numerical features
model_input_dict.update(user_input_numerical)

# Populate one-hot encoded categorical features
# Check if the selected category exists in the expected features, and set it to 1
if f'Team_{team}' in model_input_dict:
    model_input_dict[f'Team_{team}'] = 1
elif team == 'Other': # Handle 'Other' if it's not explicitly in the trained features
    # No specific one-hot column for 'Other', it will remain 0 for all specific teams
    pass
else:
    # If a selected team is not in the trained features, it will remain 0,
    # which is effectively treated as 'Other' by the model.
    pass

if f'Type_{player_type}' in model_input_dict:
    model_input_dict[f'Type_{player_type}'] = 1
elif player_type == 'Other':
    pass

if f'Batting Style_{batting_style}' in model_input_dict:
    model_input_dict[f'Batting Style_{batting_style}'] = 1
elif batting_style == 'Other':
    pass

if f'National Side_{national_side}' in model_input_dict:
    model_input_dict[f'National Side_{national_side}'] = 1
elif national_side == 'Other':
    pass

if f'Bowling_{bowling_style}' in model_input_dict:
    model_input_dict[f'Bowling_{bowling_style}'] = 1
elif bowling_style == 'Other' or bowling_style == 'No Bowling':
    pass

# Create the DataFrame for prediction, ensuring column order matches training data
input_df = pd.DataFrame([model_input_dict])
input_df = input_df[expected_features] # Reorder columns to match the training data

# --- Predict Button ---
if st.button("Predict Runs", type="primary"):
    try:
        prediction = model.predict(input_df)[0]
        # Ensure prediction is non-negative
        predicted_runs = max(0, round(prediction))
        st.success(f"🎯 Predicted Runs: **{predicted_runs}**")
        st.info("This prediction is based on the provided statistics and the trained model. Actual performance may vary.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure all inputs are valid and the model is correctly loaded.")

st.markdown("---")
