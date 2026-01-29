import os
import sys
import time
import logging

# Ensure project root is in path
sys.path.append(os.getcwd())

from utils.logger import setup_script_logger
from utils.llm_supervisor import LLMSupervisor
from dotenv import load_dotenv

# Load Env Vars
load_dotenv()

def main():
    # Setup specific logger for this service
    logger = setup_script_logger(file_name='llm_supervisor.log', level=logging.INFO)
    logger.info("🚀 Starting LLM Supervisor Microservice...")

    try:
        # Initialize Supervisor
        supervisor = LLMSupervisor()
        
        # In standalone mode, we don't start a thread, we just run the loop directly
        # But our class is designed with .start() (threaded). 
        # To keep it simple and reuse code, we'll start it and then just keep the main process alive.
        supervisor.start()
        
        while True:
            time.sleep(1)
            if not supervisor.running:
                logger.error("Supervisor thread died unexpectedly. Exiting service.")
                break
                
    except KeyboardInterrupt:
        logger.info("Service stopping by user request...")
        supervisor.stop()
    except Exception as e:
        logger.critical(f"Service crashed: {e}")
        if 'supervisor' in locals():
            supervisor.stop()
        sys.exit(1)

if __name__ == "__main__":
    main()
