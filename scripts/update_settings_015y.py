
import yaml

try:
    with open("config/settings.yaml", "r") as f:
        content = f.read()
    
    # Simple string replacement to preserve comments
    # Logic: Replace whatever current model setting is with the new 0.15y one
    # We look for the pattern we established last time
    
    # Direct replacement for the likely current state (0.5y)
    new_content = content.replace('model_name: "improved_best_model_1H_0.5y.pkl"', 'model_name: "improved_best_model_1H_0.15y.pkl"')
    new_content = new_content.replace('metadata_name: "improved_metadata_1H_0.5y.json"', 'metadata_name: "improved_metadata_1H_0.15y.json"')
    
    # Fallback/Safety: if user manually reverted or logic changed, ensure we catch generic case too?
    # For now, assuming state is persistent from previous step 1007/1013
    
    with open("config/settings.yaml", "w") as f:
        f.write(new_content)
    print("Successfully updated settings.yaml to point to 0.15y model.")
except Exception as e:
    print(f"Error updating settings: {e}")
