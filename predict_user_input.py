#!/usr/bin/env python3
"""
Interactive Parkinson's Disease Prediction Script
Allows users to input their own vocal feature data for prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import sys
import os

# Add src to path for imports
sys.path.append('src')

def load_and_train_models():
    """Load dataset and train all models"""
    print("Loading dataset and training models...")

    # Load dataset
    dataset = pd.read_csv('data/parkinsons.data')

    # Extract features (same as in original code)
    x = dataset.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23]].values
    y = dataset.iloc[:, 17].values  # status column

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Train models
    models = {
        'XGBoost': XGBClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=16, criterion="entropy", random_state=42),
        'K-Nearest Neighbor': KNeighborsClassifier(n_neighbors=8, p=2, metric='minkowski'),
        'Support Vector Machine': SVC(probability=True, random_state=42)
    }

    trained_models = {}
    print("\nTraining models...")
    for name, model in models.items():
        model.fit(x_train_scaled, y_train)
        accuracy = model.score(x_test_scaled, y_test)
        trained_models[name] = model
        print(f"{name}: {accuracy:.2%} accuracy")

    return trained_models, scaler

def get_feature_names():
    """Return the names of the 22 features used"""
    return [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
        'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
        'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    ]

def get_user_input():
    """Get vocal feature input from user"""
    features = get_feature_names()
    user_data = []

    print("\n" + "="*60)
    print("PARKINSON'S DISEASE PREDICTION - INPUT YOUR VOCAL FEATURES")
    print("="*60)
    print("Please enter the following vocal feature measurements:")
    print("(You can get these from voice analysis software)")
    print("-"*60)

    for i, feature in enumerate(features, 1):
        while True:
            try:
                value = float(input(f"{i:2d}. {feature}: "))
                user_data.append(value)
                break
            except ValueError:
                print("    Please enter a valid number.")

    return np.array(user_data).reshape(1, -1)

def predict_with_models(user_data, models, scaler):
    """Make predictions using all trained models"""
    # Scale the user input
    user_data_scaled = scaler.transform(user_data)

    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)

    predictions = {}
    for name, model in models.items():
        prediction = model.predict(user_data_scaled)[0]
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(user_data_scaled)[0]
            prob_healthy = probability[0] * 100
            prob_parkinsons = probability[1] * 100
        else:
            prob_healthy = prob_parkinsons = "N/A"

        predictions[name] = {
            'prediction': prediction,
            'prob_healthy': prob_healthy,
            'prob_parkinsons': prob_parkinsons
        }

        result = "PARKINSON'S DISEASE DETECTED" if prediction == 1 else "HEALTHY"
        print(f"\n{name}:")
        print(f"  Prediction: {result}")
        if prob_healthy != "N/A":
            print(f"  Confidence - Healthy: {prob_healthy:.1f}%")
            print(f"  Confidence - Parkinson's: {prob_parkinsons:.1f}%")

    return predictions

def show_sample_data():
    """Show sample data to help users understand the input format"""
    print("\n" + "="*60)
    print("SAMPLE DATA (for reference)")
    print("="*60)

    # Load a sample from the dataset
    dataset = pd.read_csv('data/parkinsons.data')
    sample = dataset.iloc[0]  # First row
    features = get_feature_names()

    print("Here's an example of vocal feature values from the dataset:")
    print("-"*60)

    for i, feature in enumerate(features):
        # Map feature names to dataset columns
        col_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 14, 15, 16, 18, 19, 20, 21, 22, 23]
        value = sample.iloc[col_indices[i]]
        print(f"{i+1:2d}. {feature}: {value}")

    status = "Parkinson's Disease" if sample.iloc[17] == 1 else "Healthy"
    print(f"\nActual diagnosis: {status}")

def main():
    """Main function"""
    print("Parkinson's Disease Detection System")
    print("Based on Vocal Feature Analysis")
    print("="*60)

    while True:
        print("\nOptions:")
        print("1. Make a prediction with your vocal features")
        print("2. View sample data format")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == '1':
            try:
                # Load and train models
                models, scaler = load_and_train_models()

                # Get user input
                user_data = get_user_input()

                # Make predictions
                predictions = predict_with_models(user_data, models, scaler)

                # Summary
                print("\n" + "="*60)
                print("SUMMARY")
                print("="*60)
                parkinsons_count = sum(1 for p in predictions.values() if p['prediction'] == 1)
                total_models = len(predictions)

                if parkinsons_count > total_models / 2:
                    print("⚠️  MAJORITY OF MODELS INDICATE PARKINSON'S DISEASE")
                    print("   Please consult with a medical professional for proper diagnosis.")
                else:
                    print("✅ MAJORITY OF MODELS INDICATE HEALTHY")
                    print("   However, this is not a substitute for professional medical advice.")

                print(f"\nModels indicating Parkinson's: {parkinsons_count}/{total_models}")

            except FileNotFoundError:
                print("Error: Dataset file not found. Make sure 'data/parkinsons.data' exists.")
            except Exception as e:
                print(f"Error: {e}")

        elif choice == '2':
            try:
                show_sample_data()
            except FileNotFoundError:
                print("Error: Dataset file not found. Make sure 'data/parkinsons.data' exists.")
            except Exception as e:
                print(f"Error: {e}")

        elif choice == '3':
            print("Thank you for using the Parkinson's Disease Detection System!")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
