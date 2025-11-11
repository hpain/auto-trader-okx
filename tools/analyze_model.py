# tools/analyze_model.py
import os
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import joblib
import yaml

def analyze_model(model_path: str, config_path: str, out_dir: str):
    """
    Analyzes the feature importance of a trained LightGBM model.

    Args:
        model_path (str): Path to the saved model file (.pkl).
        config_path (str): Path to the settings.yaml file.
        out_dir (str): Directory to save the feature importance plot.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load the model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if not isinstance(model, lgb.LGBMClassifier):
        print(f"Error: Loaded object is not a LightGBM Classifier model. It is {type(model)}")
        return

    # Create feature importance plot
    fig, ax = plt.subplots(figsize=(12, 8))
    lgb.plot_importance(model, ax=ax, max_num_features=30)
    plt.title("Feature Importance of the Best Model")
    plt.tight_layout()

    # Save the plot
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    plot_path = os.path.join(out_dir, "latest_model_feature_importance.png")
    plt.savefig(plot_path)
    print(f"Feature importance plot saved to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    # Assuming the script is run from the root directory of the project
    MODEL_FILE = "models/best_model.pkl"
    CONFIG_FILE = "config/settings.yaml"
    OUTPUT_DIR = "results"
    
    analyze_model(MODEL_FILE, CONFIG_FILE, OUTPUT_DIR)
