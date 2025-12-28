import yaml
import os

SETTINGS_PATH = 'config/settings.yaml'

def update_settings():
    if not os.path.exists(SETTINGS_PATH):
        print(f"Error: {SETTINGS_PATH} not found.")
        return

    with open(SETTINGS_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Add Risk Engine Config if not present
    if 'risk_management' not in config:
        config['risk_management'] = {}

    rm = config['risk_management']
    
    # Update/Add keys
    rm['use_risk_engine'] = True
    rm['target_volatility'] = 0.20
    rm['volatility_window'] = 30
    
    # Ensure other keys exist
    if 'risk_per_trade' not in rm: rm['risk_per_trade'] = 0.02
    if 'max_portfolio_risk' not in rm: rm['max_portfolio_risk'] = 0.10
    if 'stop_loss_pct' not in rm: rm['stop_loss_pct'] = 0.02
    if 'take_profit_pct' not in rm: rm['take_profit_pct'] = 0.04

    with open(SETTINGS_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Successfully updated {SETTINGS_PATH} with Risk Engine settings.")

if __name__ == "__main__":
    update_settings()
