import logging
import sys
from pathlib import Path

class CustomLogger:
    
    def __init__(self, name: str, console: bool = True, level: int = logging.INFO, show_logs: bool = True):
        """
        Create a logger with a given name.
        - Always writes to log/{name}.log
        - Optionally prints to console
        - Supports multiple independent loggers
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # remember show_logs preference
        self.show_logs = show_logs

        # If logger has no handlers, configure it
        if not self.logger.handlers:
            log_dir = Path("log")
            log_dir.mkdir(parents=True, exist_ok=True)

            formatter = logging.Formatter(
                "[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )

            # File handler (unique per logger name)
            file_handler = logging.FileHandler(log_dir / f"{name}.log", mode="a", encoding="utf-8")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # Console handler (optional) and controlled by show_logs
            if console and self.show_logs:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(formatter)
                console_handler.set_name(f"console-{name}")
                self.logger.addHandler(console_handler)
        else:
            # Logger already exists: ensure console handler presence matches show_logs
            # If show_logs is True and console requested, add a console handler if missing
            if console and self.show_logs:
                has_console = any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers)
                if not has_console:
                    console_handler = logging.StreamHandler(sys.stdout)
                    console_handler.setFormatter(formatter)
                    console_handler.set_name(f"console-{name}")
                    self.logger.addHandler(console_handler)
            # If show_logs is False, remove any StreamHandler so terminal output is suppressed
            if not self.show_logs:
                # remove StreamHandlers (console handlers)
                self.logger.handlers = [h for h in self.logger.handlers if not isinstance(h, logging.StreamHandler)]

    def get_logger(self):
        return self.logger