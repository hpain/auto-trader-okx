import shutil
import os
import datetime
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/backup.log"),
        logging.StreamHandler()
    ]
)

DB_FILE = "trader_state.db"
BACKUP_DIR = "backups"

def backup_database():
    """
    Backs up the SQLite database to the backup directory.
    Retains backups for 7 days.
    """
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
        logging.info(f"Created backup directory: {BACKUP_DIR}")

    if not os.path.exists(DB_FILE):
        logging.warning(f"Database file {DB_FILE} not found. Skipping backup.")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(BACKUP_DIR, f"trader_state_{timestamp}.db")

    try:
        # Use sqlite3 API to backup if possible (safer for active DBs), 
        # but file copy is often sufficient for WAL mode if checkpointed.
        # For simplicity and robustness with WAL, we use shutil.copy2
        # Ideally, one should run 'PRAGMA wal_checkpoint(TRUNCATE);' before copy if strict consistency is needed,
        # but for this use case, a hot backup is usually acceptable or we can rely on WAL file being copied too if needed.
        # Note: shutil.copy2 might not copy the -wal and -shm files if they exist.
        # A better approach for hot backup is using the sqlite3 backup API.
        
        import sqlite3
        
        src = sqlite3.connect(DB_FILE)
        dst = sqlite3.connect(backup_file)
        
        with src, dst:
            src.backup(dst)
            
        src.close()
        dst.close()
        
        logging.info(f"Database backed up successfully to {backup_file}")
        
        # Cleanup old backups
        cleanup_old_backups()
        
    except Exception as e:
        logging.error(f"Backup failed: {e}")

def cleanup_old_backups(days_to_keep=7):
    """
    Removes backups older than days_to_keep.
    """
    now = time.time()
    cutoff = now - (days_to_keep * 86400)
    
    for filename in os.listdir(BACKUP_DIR):
        file_path = os.path.join(BACKUP_DIR, filename)
        if os.path.isfile(file_path):
            if os.path.getmtime(file_path) < cutoff:
                try:
                    os.remove(file_path)
                    logging.info(f"Deleted old backup: {filename}")
                except Exception as e:
                    logging.error(f"Failed to delete {filename}: {e}")

if __name__ == "__main__":
    logging.info("Starting database backup...")
    backup_database()
    logging.info("Backup process completed.")
