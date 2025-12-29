import pickle
import os
import sys

def inspect_scaler(path="models/scaler_long_term.pkl"):
    print(f"Inspecting scaler at: {path}")
    if not os.path.exists(path):
        print("Error: File not found.")
        return

    try:
        with open(path, "rb") as f:
            scaler = pickle.load(f)
        
        print("\n--- Scaler Attributes ---")
        print(f"Type: {type(scaler)}")
        
        if hasattr(scaler, 'feature_names_in_'):
            print(f"✅ feature_names_in_ found! Count: {len(scaler.feature_names_in_)}")
            print(f"First 5 features: {scaler.feature_names_in_[:5]}")
        else:
            print("❌ feature_names_in_ NOT found.")

        if hasattr(scaler, 'n_features_in_'):
            print(f"✅ n_features_in_ found: {scaler.n_features_in_}")
        else:
            print("❌ n_features_in_ NOT found.")
            
        print("-" * 30)

    except Exception as e:
        print(f"Error loading scaler: {e}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "models/scaler_long_term.pkl"
    inspect_scaler(path)
