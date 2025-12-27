
import yaml

try:
    with open("config/settings.yaml", "r") as f:
        content = f.read()
    
    # Simple string replacement to preserve comments
    new_content = content.replace('model_name: "best_model.pkl"', 'model_name: "improved_best_model_1H_0.5y.pkl"')
    new_content = new_content.replace('metadata_name: "metadata.json"', 'metadata_name: "improved_metadata_1H_0.5y.json"')
    
    with open("config/settings.yaml", "w") as f:
        f.write(new_content)
    print("Successfully updated settings.yaml")
except Exception as e:
    print(f"Error updating settings: {e}")
