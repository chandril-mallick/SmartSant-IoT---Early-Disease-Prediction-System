"""
Quick Reference: Running Model Evaluation

This script demonstrates how to use the evaluation functions
for different scenarios and model types.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.evaluate_model import (
    evaluate_model,
    compute_specificity,
    plot_roc_curve,
    plot_confusion_matrix
)
import numpy as np


# Example 1: Evaluate with true labels and predictions
def example_basic_evaluation():
    """Basic evaluation with predictions"""
    # Simulated data
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.8, 0.6, 0.2, 0.7, 0.9])
    
    results = evaluate_model(
        y_true=y_true,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        model_name="Example Model"
    )
    
    return results


# Example 2: Compute specificity manually
def example_specificity():
    """Compute specificity from predictions"""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 0, 1, 1])
    
    specificity = compute_specificity(y_true, y_pred)
    print(f"Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
    
    return specificity


# Example 3: Using scikit-learn models
def example_sklearn_model():
    """Evaluate scikit-learn model"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=2,
        class_sep=1.0,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    results = evaluate_model(
        y_true=y_test,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        model_name="Scikit-learn Logistic Regression"
    )
    
    return results


# Example 4: Evaluate with custom threshold
def example_custom_threshold():
    """Evaluate with custom classification threshold"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Get probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Try different thresholds
    thresholds = [0.3, 0.5, 0.7]
    
    print("\n" + "="*60)
    print("Evaluation at Different Thresholds")
    print("="*60)
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        print(f"\n\nThreshold: {threshold}")
        results = evaluate_model(
            y_true=y_test,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            model_name=f"Model (threshold={threshold})"
        )


# Example 5: Save plots
def example_save_plots():
    """Generate and save evaluation plots"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    results = evaluate_model(
        y_true=y_test,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        model_name="Example Model with Plots"
    )
    
    # Save plots
    os.makedirs('example_plots', exist_ok=True)
    plot_roc_curve(y_test, y_pred_proba, save_path='example_plots/roc_curve.png')
    plot_confusion_matrix(results['confusion_matrix'], save_path='example_plots/confusion_matrix.png')
    
    print("\nPlots saved to 'example_plots/' directory")


if __name__ == "__main__":
    print("Model Evaluation Examples\n")
    print("Select an example to run:")
    print("1. Basic evaluation")
    print("2. Compute specificity")
    print("3. Evaluate sklearn model")
    print("4. Custom threshold evaluation")
    print("5. Save evaluation plots")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        example_basic_evaluation()
    elif choice == '2':
        example_specificity()
    elif choice == '3':
        example_sklearn_model()
    elif choice == '4':
        example_custom_threshold()
    elif choice == '5':
        example_save_plots()
    else:
        print("Invalid choice. Running all examples...")
        example_basic_evaluation()
        example_specificity()
        example_sklearn_model()
