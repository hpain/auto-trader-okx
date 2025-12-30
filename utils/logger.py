import logging
from logging.handlers import RotatingFileHandler
import json
from pathlib import Path

def setup_script_logger(log_dir: str = 'logs', file_name: str = 'script.log', level=logging.INFO):
    """
    Set up a basic logger for scripts that logs to both console and a file.
    This configures the root logger.
    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    # Custom Colored Formatter
    # Custom Colored Formatter
    class ColoredFormatter(logging.Formatter):
        """Custom formatter to add colors to console logs"""
        
        # ANSI Escape Codes
        # Debug/Info(Default) -> Grey (Dim)
        GREY = "\033[38;5;250m" # Light Grey
        DIM_GREY = "\033[38;5;240m" # Darker Grey for Debug
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        RED = "\033[31m"
        BOLD_RED = "\033[1;31m"
        RESET = "\033[0m"

        FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

        # Base formats without conditional INFO
        FORMATS = {
            logging.DEBUG: DIM_GREY + FORMAT + RESET,
            logging.WARNING: YELLOW + FORMAT + RESET,
            logging.ERROR: RED + FORMAT + RESET,
            logging.CRITICAL: BOLD_RED + FORMAT + RESET
        }

        def format(self, record):
            if record.levelno == logging.INFO:
                msg = record.msg if isinstance(record.msg, str) else str(record.msg)
                # Success/Positive keywords -> GREEN
                if any(k in msg for k in ["Successfully", "OK:", "Complete", "Loaded", "Saved"]):
                     log_fmt = self.GREEN + self.FORMAT + self.RESET
                else:
                     # Normal Info -> Light Grey
                     log_fmt = self.GREY + self.FORMAT + self.RESET
            else:
                log_fmt = self.FORMATS.get(record.levelno, self.RESET + self.FORMAT + self.RESET)
                
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)

    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = ColoredFormatter()

    # File handler
    file_handler = logging.FileHandler(log_path / file_name)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

# Initialize a default logger for modules to import
logger = setup_script_logger()

def setup_trader_logger(log_dir: str = 'logs', file_name: str = 'trading_cycles.log', max_bytes: int = 10*1024*1024, backup_count: int = 5):
    """
    配置并返回一个用于记录结构化JSON交易周期的logger。

    :param log_dir: 存放日志文件的目录。
    :param file_name: 日志文件的名称。
    :param max_bytes: 每个日志文件的最大大小。
    :param backup_count: 保留的旧日志文件数量。
    :return: 配置好的logger实例。
    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    trader_logger = logging.getLogger('TraderLogger')
    trader_logger.setLevel(logging.INFO)
    
    # 防止重复添加handler
    if trader_logger.hasHandlers():
        trader_logger.handlers.clear()

    # 创建一个只处理消息的handler，不做任何格式化
    handler = RotatingFileHandler(log_path / file_name, maxBytes=max_bytes, backupCount=backup_count)
    
    # 我们直接记录JSON字符串，所以formatter不是必需的，但为了完整性可以定义一个
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    
    trader_logger.addHandler(handler)
    
    return trader_logger

class CycleLogger:
    """
    一个辅助类，用于在单个交易周期内收集、组织和记录日志信息。
    """
    def __init__(self, logger: logging.Logger, cycle_id: str):
        self.logger = logger
        self.cycle_id = cycle_id
        self.log_data = {
            "cycle_id": self.cycle_id,
            "status": "PENDING",
            "decision": {},
            "execution": {},
            "portfolio": {},
            "error": None
        }

    def set_status(self, status: str):
        self.log_data['status'] = status

    def add_info(self, message: str):
        """Generic info logging for the cycle."""
        if 'info' not in self.log_data:
            self.log_data['info'] = []
        self.log_data['info'].append(message)
        # Also log to standard logger for immediate visibility
        self.logger.info(f"[Cycle {self.cycle_id}] {message}")

    def add_warning(self, message: str):
        """Generic warning logging for the cycle."""
        if 'warnings' not in self.log_data:
            self.log_data['warnings'] = []
        self.log_data['warnings'].append(message)
        self.logger.warning(f"[Cycle {self.cycle_id}] {message}")

    def add_error(self, message: str):
        """Generic error logging for the cycle."""
        if 'errors' not in self.log_data:
            self.log_data['errors'] = []
        self.log_data['errors'].append(message)
        self.logger.error(f"[Cycle {self.cycle_id}] {message}")

    def add_decision_info(self, **kwargs):
        self.log_data['decision'].update(kwargs)

    def add_execution_info(self, **kwargs):
        self.log_data['execution'].update(kwargs)

    def add_portfolio_info(self, **kwargs):
        self.log_data['portfolio'].update(kwargs)

    def set_error(self, error_message: str):
        self.log_data['status'] = 'FAILED'
        self.log_data['error'] = error_message

    def commit(self):
        """
        将收集到的日志信息作为一条JSON记录提交。
        """
        # 确保在最终提交前状态不是PENDING
        if self.log_data['status'] == 'PENDING':
            self.log_data['status'] = 'SUCCESS' if self.log_data['error'] is None else 'FAILED'
            
        log_string = json.dumps(self.log_data)
        self.logger.info(log_string)