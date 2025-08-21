# 🏏 IPL Cricket Batting Score Predictor

## Project Overview

This project is an end-to-end machine learning solution designed to predict the batting runs a player is likely to score in an Indian Premier League (IPL) match. Leveraging historical IPL player statistics, the system cleans and processes raw data, trains a predictive model, and provides an interactive web application for users to input player stats and receive immediate run predictions.

This serves as a practical demonstration of data preprocessing, feature engineering, model training, and web application deployment using Python's popular ML and web development libraries.

## Features

* **Data Preprocessing:** Handles raw IPL data, including cleaning missing values, converting string-formatted numerical data (e.g., 'Age', 'HighestInnScore'), and preparing it for model training.
* **Feature Engineering:** Transforms categorical features (like 'Team', 'Type', 'Batting Style', 'National Side', 'Bowling') into numerical representations using one-hot encoding.
* **Model Training:** Utilizes a **Random Forest Regressor** to learn patterns from the processed data and make accurate run predictions.
* **Model Persistence:** Saves the trained model and the exact feature set for consistent use in the prediction application.
* **Interactive Web Application:** A user-friendly interface built with **Streamlit** allows users to input various player statistics and get real-time run predictions.

## Technologies Used

* **Python 3.x**
* **pandas:** For data manipulation and analysis.
* **scikit-learn:** For machine learning model training (RandomForestRegressor, train_test_split, metrics).
* **streamlit:** For building the interactive web application.
* **numpy:** For numerical operations (used implicitly by pandas/sklearn).
* **matplotlib** and **seaborn:** (Optional, for visualizing correlation heatmap)

## Dataset

The project utilizes `IPL_Data.csv`, a dataset containing various statistics for IPL players, including their batting performance, team, type, age, and other relevant metrics. The dataset undergoes cleaning to handle missing values and filter out irrelevant entries (e.g., players with 0 Cr value).

## Model

The core of the prediction system is a **Random Forest Regressor**. This ensemble learning method is well-suited for regression tasks, capable of handling both numerical and categorical features, and provides good predictive power while being relatively robust to outliers and multicollinearity.

## Setup and Installation

To set up and run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install the required Python packages:**
    ```bash
    pip install pandas scikit-learn streamlit numpy matplotlib seaborn
    ```
4.  **Place your dataset:**
    Ensure your `IPL_Data.csv` file is placed in the root directory of the project, next to `train_model.py` and `app.py`.

## How to Run

Follow these two steps to train the model and launch the prediction application:

1.  **Train the Machine Learning Model:**
    This script will clean the data, train the Random Forest Regressor, and save the trained model (along with the feature columns) as `cricket_model.pkl`.
    ```bash
    python train_model.py
    ```
    You should see output indicating the model's performance (R2 score, MSE) and confirmation of the model being saved.

2.  **Run the Streamlit Web Application:**
    This will launch the interactive prediction interface in your web browser.
    ```bash
    streamlit run app.py
    ```
    Your default web browser should open automatically, displaying the Streamlit application. You can now input player statistics and get real-time run predictions.

## Project Structure
