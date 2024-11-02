import matplotlib.pyplot as plt  # Used for creating plots
import seaborn as sns  # Used for visualizing data, especially residuals
from sklearn.model_selection import learning_curve  # Used for generating learning curve data
import numpy as np  # Used for numerical operations


def plot_learning_curve(model, X, y):
    """
    Plots the learning curve for a given model.

    Parameters:
    - model: The machine learning model to evaluate.
    - X: Training data features.
    - y: Training data labels/target variable.

    This function calculates and plots the mean squared error (MSE) for the training 
    and validation sets at different training sizes, allowing for the evaluation 
    of model performance and potential overfitting or underfitting.
    """
    # Generate learning curve data with MSE as the metric
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
    )

    # Compute the mean training and validation errors
    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)

    # Plot the learning curve
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, label='Training error')
    plt.plot(train_sizes, test_scores_mean, label='Validation error')
    plt.ylabel('MSE')
    plt.xlabel('Training size')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()


def plot_residuals(y_true, y_pred):
    """
    Plots the residuals of predictions to assess model errors.

    Parameters:
    - y_true: The true values of the target variable.
    - y_pred: The predicted values from the model.

    This function calculates and plots residuals (the difference between actual 
    and predicted values) against the predicted values, providing insights into 
    model accuracy and identifying patterns in prediction errors.
    """
    # Calculate residuals (errors)
    residuals = y_true - y_pred

    # Plot residuals
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')  # Horizontal line at 0 for reference
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()


def plot_important_features(model, preprocessed_train_df):
    """
    Plots the most important features in predicting trip duration.

    Parameters:
    - model: A trained machine learning model with feature_importances_ attribute.
    - preprocessed_train_df: DataFrame of preprocessed training data features.

    This function identifies the top N most important features (excluding those
    with "line_id" in their name) based on the model’s feature importance values 
    and visualizes them in a bar chart to highlight the key drivers in predictions.
    """
    # Extract feature names and importance, excluding "line_id" features
    feature_names = preprocessed_train_df.columns
    important_features = [(feature, importance) for feature, importance in
                          zip(feature_names, model.feature_importances_) if "line_id" not in feature]
    # Sort features by importance
    important_features = sorted(important_features, key=lambda x: x[1], reverse=True)

    # Select top features for plotting
    top_n = 10  # Number of top features to display
    top_features = important_features[:top_n]

    # Plot the top features
    plt.figure(figsize=(10, 6))
    plt.bar([f[0] for f in top_features], [f[1] for f in top_features], align="center")
    plt.xticks(rotation=90)
    plt.title("Top Feature Importances in Trip Duration Prediction (Excluding Line IDs)")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()
