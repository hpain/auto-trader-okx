
import pickle
import pandas as pd
import lightgbm as lgb
import os

model_path = r"models/improved_best_model_1H_3.0y.pkl"

def inspect():
    if not os.path.exists(model_path):
        print("Model not found")
        return

    print(f"Loading {model_path}...")
    import joblib
    with open(model_path, 'rb') as f:
        data = joblib.load(f)
    
    # Check if data is just the model or a dict
    if isinstance(data, dict):
        model = data.get('model')
        if not model:
            print("Keys:", data.keys())
            model = data['best_trial_params'] # Might vary
    else:
        model = data

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = model.feature_name_
        
        df = pd.DataFrame(sorted(zip(importances, feature_names)), columns=['Value','Feature'])
        print("\n--- TOP 20 FEATURES ---")
        print(df.tail(20).to_string())
        
        print("\n--- BOTTOM 20 FEATURES ---")
        print(df.head(20).to_string())
    else:
        print("Model object structure unclear:", type(model))
        # Try to inspect if it is raw Booster
        if isinstance(model, lgb.Booster):
             print("LGB Booster detected")
             print(model.feature_importance())

if __name__ == "__main__":
    inspect()
