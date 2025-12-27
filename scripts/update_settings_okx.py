
import yaml

try:
    with open("config/settings.yaml", "r") as f:
        content = f.read()
    
    # Replace primary_exchange: 'binance' with 'okx'
    # Be careful not to break other things
    # We target the specific lines around verify location
    
    if "primary_exchange: 'binance'" in content:
        new_content = content.replace("primary_exchange: 'binance'", "primary_exchange: 'okx'")
        with open("config/settings.yaml", "w") as f:
            f.write(new_content)
        print("Successfully switched primary_exchange to 'okx'")
    elif "primary_exchange: 'okx'" in content:
        print("primary_exchange is already 'okx'")
    else:
        print("Could not find primary_exchange setting to update.")

except Exception as e:
    print(f"Error updating settings: {e}")
