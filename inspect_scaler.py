import pickle
import pandas as pd
import sys

def inspect_scaler():
    try:
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        print(f"Scaler Type: {type(scaler)}")
        if hasattr(scaler, 'n_features_in_'):
            print(f"Num Features: {scaler.n_features_in_}")
            
        if hasattr(scaler, 'feature_names_in_'):
            feats = scaler.feature_names_in_.tolist()
            print("Feature Names (First 10):")
            print(feats[:10])
            print("\nFeature Names (Last 10):")
            print(feats[-10:])
            
            # Save to file for easy reading
            pd.DataFrame(feats, columns=['feature']).to_csv('model_features.csv', index=False)
            print("\nSaved full feature list to model_features.csv")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_scaler()
