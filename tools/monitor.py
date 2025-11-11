import time
import os
from datetime import datetime, timedelta

# --- Configuration ---
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'logs', 'trader.log')
CHECK_INTERVAL_SECONDS = 60  # Check every 60 seconds
ERROR_KEYWORDS = ["ERROR", "CRITICAL", "Traceback"]
TIME_WINDOW_MINUTES = 5 # Only check for errors in the last 5 minutes

def check_log_for_errors():
    """Checks the log file for recent errors."""
    if not os.path.exists(LOG_FILE_PATH):
        print(f"[{datetime.now()}] ALARM: Log file not found at {LOG_FILE_PATH}!")
        return

    try:
        with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"[{datetime.now()}] ALARM: Could not read log file: {e}")
        return

    recent_errors = []
    time_window = datetime.now() - timedelta(minutes=TIME_WINDOW_MINUTES)

    for line in lines:
        try:
            # Assuming log format: "YYYY-MM-DD HH:MM:SS,ms - LEVEL - ..."
            log_time_str = line.split(' - ')[0]
            log_time = datetime.strptime(log_time_str, '%Y-%m-%d %H:%M:%S,%f')

            if log_time > time_window:
                if any(keyword in line for keyword in ERROR_KEYWORDS):
                    recent_errors.append(line.strip())
        except (ValueError, IndexError):
            # Ignore lines that don't match the expected format
            continue

    if recent_errors:
        print(f"\n{'='*20} ALARM TRIGGERED AT {datetime.now()} {'='*20}")
        print(f"Found {len(recent_errors)} recent error(s) in the last {TIME_WINDOW_MINUTES} minutes:")
        for error_line in recent_errors:
            print(f"- {error_line}")
        print("ACTION: Check the main application immediately!")
        print('=' * (42 + len(str(datetime.now()))))
        # TODO: Add your notification logic here, e.g.:
        # send_telegram_alert("Trader Bot Error Detected!")
        # send_email_alert("Error in Trader Bot", "\n".join(recent_errors))
    else:
        print(f"[{datetime.now()}] OK: No recent errors found.")

def main():
    """Main loop to run the monitor continuously."""
    print("--- Trader Log Monitor Started ---")
    print(f"Monitoring log file: {LOG_FILE_PATH}")
    print(f"Checking every {CHECK_INTERVAL_SECONDS} seconds for keywords: {ERROR_KEYWORDS}")
    print("------------------------------------")
    while True:
        check_log_for_errors()
        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitor stopped by user.")
