import os
import sys
import subprocess
import argparse
import datetime
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd):
    """Runs a shell command and checks for errors."""
    logger.info(f"Executing: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Auto-Evolution: Download Data -> Retrain Model")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair symbol")
    parser.add_argument("--lookback_years", type=int, default=2, help="How many years of history to use")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--deploy", action="store_true", help="Attempt to upload to cloud (requires SCP config)")
    parser.add_argument("--cloud_host", type=str, default="user@your-cloud-ip", help="Cloud SSH host (e.g. root@1.2.3.4)")
    parser.add_argument("--cloud_path", type=str, default="/root/auto-trader-okx/models/", help="Remote models directory")
    
    args = parser.parse_args()
    
    # 1. Determine Date Range
    now = datetime.datetime.now()
    end_year = now.year
    start_year = end_year - args.lookback_years + 1 # e.g. 2025 - 2 + 1 = 2024. Range: 2024, 2025.
    
    logger.info(f"--- Starting Evolution Cycle for {args.symbol} ({start_year}-{end_year}) ---")
    
    # 2. Fetch Data (The Fuel)
    # Output format from fetch_full_dataset is: data/history/{symbol}_FULL_{start}_{end}.csv
    # We call the script
    fetch_script = os.path.join("scripts", "fetch_full_dataset.py")
    cmd_fetch = [
        sys.executable, fetch_script,
        "--symbol", args.symbol,
        "--start_year", str(start_year),
        "--end_year", str(end_year)
    ]
    
    if not run_command(cmd_fetch):
        logger.error("Data fetching failed. Aborting evolution.")
        sys.exit(1)
        
    # Construct expected filename to pass to trainer
    # logic in fetch_full_dataset: f"{args.output_dir}/{args.symbol}_FULL_{args.start_year}_{args.end_year}.csv"
    data_file = f"data/history/{args.symbol}_FULL_{start_year}_{end_year}.csv"
    
    if not os.path.exists(data_file):
        logger.error(f"Expected data file not found: {data_file}")
        sys.exit(1)
        
    logger.info(f"Data ready: {data_file}")
    
    # 3. Retrain Model (The Sword)
    train_script = os.path.join("scripts", "train_transformer.py")
    model_output = "models/transformer_v1.pth"
    
    cmd_train = [
        sys.executable, train_script,
        "--data", data_file,
        "--model_path", model_output,
        "--epochs", str(args.epochs)
    ]
    
    if not run_command(cmd_train):
        logger.error("Training failed. Aborting evolution.")
        sys.exit(1)
        
    logger.info(f"Model successfully trained: {model_output}")
    
    # 4. Deploy (The Bridge)
    if args.deploy:
        logger.info(f"Attempting deployment to {args.cloud_host}:{args.cloud_path}...")
        # Note: This assumes SSH keys are set up. If password is needed, this will fail/hang.
        cmd_scp = ["scp", model_output, f"{args.cloud_host}:{args.cloud_path}"]
        if run_command(cmd_scp):
            logger.info("✅ Deployment Successful! The Cloud Bot should pick up the new model on restart.")
        else:
            logger.error("Deployment failed. Please check your SSH config or upload manually.")
    else:
        logger.info("--- Evolution Complete (Local Only) ---")
        logger.info(f"New model is at: {os.path.abspath(model_output)}")
        logger.info("To deploy manually:")
        logger.info(f"  scp {model_output} root@<YOUR_SERVER_IP>:/path/to/models/")

if __name__ == "__main__":
    main()
