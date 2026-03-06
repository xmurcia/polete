"""
Error Logger - Sistema centralizado de logging de errores
Guarda errores en archivo con timestamp y contexto
"""

import os
import traceback
from datetime import datetime
from pathlib import Path


class ErrorLogger:
    """Logger centralizado para errores del bot"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.error_log = self.log_dir / "errors.log"

    def log_error(self, error: Exception, context: str = "", extra_info: dict = None):
        """
        Log an error with full traceback and context

        Args:
            error: Exception object
            context: Description of what was happening when error occurred
            extra_info: Additional context (dict)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_type = type(error).__name__
        error_msg = str(error)

        # Get full traceback
        tb = traceback.format_exc()

        # Build log entry
        log_entry = [
            "=" * 80,
            f"TIMESTAMP: {timestamp}",
            f"ERROR TYPE: {error_type}",
            f"ERROR MESSAGE: {error_msg}",
        ]

        if context:
            log_entry.append(f"CONTEXT: {context}")

        if extra_info:
            log_entry.append("EXTRA INFO:")
            for key, value in extra_info.items():
                log_entry.append(f"  {key}: {value}")

        log_entry.extend([
            "TRACEBACK:",
            tb,
            "=" * 80,
            ""  # Empty line for readability
        ])

        # Write to file
        with open(self.error_log, "a", encoding="utf-8") as f:
            f.write("\n".join(log_entry))

        print(f"[ErrorLogger] ❌ Error logged to {self.error_log}")

    def log_simple(self, message: str, level: str = "ERROR"):
        """
        Log a simple message without traceback

        Args:
            message: Message to log
            level: Log level (ERROR, WARNING, INFO)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_entry = f"[{timestamp}] [{level}] {message}\n"

        with open(self.error_log, "a", encoding="utf-8") as f:
            f.write(log_entry)

    def get_recent_errors(self, lines: int = 50) -> str:
        """Get last N lines from error log"""
        if not self.error_log.exists():
            return "No errors logged yet"

        with open(self.error_log, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
            return "".join(all_lines[-lines:])

    def clear_old_errors(self, days: int = 7):
        """Archive errors older than N days (optional maintenance)"""
        # TODO: Implement rotation if needed
        pass


# Global instance
_error_logger = None


def get_error_logger() -> ErrorLogger:
    """Get global error logger instance"""
    global _error_logger
    if _error_logger is None:
        _error_logger = ErrorLogger()
    return _error_logger
