from utils.preprocessing import load_and_preprocess_data
from models.model import build_and_train_models
from utils.prediction import predict_dropout
from utils.visualization import plot_feature_distributions
import pandas as pd
import numpy as np

def get_user_input():
    school = input("Enter the school name: ")
    area = input("Enter the area (Urban/Rural): ")
    gender = input("Enter gender (Male/Female): ")
    caste = input("Enter caste: ")
    age = int(input("Enter age: "))
    religion = input("Enter religion: ")
    annual_family_income = int(input("Enter annual family income: "))
    
    return {
        'school': school,
        'area': area,
        'gender': gender,
        'caste': caste,
        'age': age,
        'religion': religion,
        'annual_family_income': annual_family_income
    }

def get_predictions(model, X_test, label_encoders, scaler):
    # Make predictions using the model
    predictions = model.predict(X_test)
    
    # Convert predictions to readable format (for neural network)
    predictions = [1 if pred > 0.5 else 0 for pred in predictions]  # For neural network

    # Decode the predicted values for categorical features (if required)
    predictions_df = pd.DataFrame(X_test, columns=['school', 'area', 'gender', 'caste', 'age', 'religion', 'annual_family_income'])
    predictions_df['dropout'] = predictions
    return predictions_df

if __name__ == "__main__":
    # Load and preprocess the data
    (X_train, X_test, y_train, y_test), scaler, label_encoders = load_and_preprocess_data("E:\Programs\school-dropout-analysis\data\dataset.csv")
    
    # Train models and select the best one
    best_model, model_accuracies = build_and_train_models(X_train, y_train, X_test, y_test)

    # Show model accuracies comparison
    print("\nModel Accuracy Comparison:")
    for model_name, accuracy in model_accuracies.items():
        print(f"{model_name}: {accuracy:.4f}")
    
    # Get predictions for the test set
    predictions_df = get_predictions(best_model, X_test, label_encoders, scaler)
    
    # Visualize the comparison of actual vs predicted data
    print("\nVisualizing feature distributions with actual vs predicted data...")
    plot_feature_distributions(X_test, predictions_df)
    
    # Get user input and predict dropout status
    print("\nPlease enter the following details to predict dropout status:")
    user_input = get_user_input()
    
    result = predict_dropout(user_input, best_model, scaler, label_encoders)
    print("Predicted dropout status:", result)