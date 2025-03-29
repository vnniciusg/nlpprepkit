import time
import logging
from functools import wraps
from typing import Callable, Dict, List, Any


def setup_logging(logger_name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging for a given logger name.

    Args:
        logger_name (str): The name of the logger.
        log_level (str): The logging level. Default is "INFO".

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


class ProfilingDecorator:
    """Decorator para coletar métricas de performance em funções de processamento"""

    def __init__(self):
        self.profile_data: List[Dict[str, Any]] = []

    def __call__(self, func: Callable[[str], str]) -> Callable[[str], str]:
        """Método principal do decorador"""

        @wraps(func)
        def wrapper(text: str) -> str:
            start_time = time.perf_counter()
            result = func(text)
            elapsed = time.perf_counter() - start_time

            self.record_metrics(func.__name__, elapsed, len(text), len(result))

            return result

        return wrapper

    def record_metrics(self, step_name: str, time: float, input_len: int, output_len: int) -> None:
        """Armazena as métricas coletadas"""
        self.profile_data.append({"step": step_name, "time": time, "input_length": input_len, "output_length": output_len})

    def get_summary(self) -> Dict[str, Any]:
        """Gera um resumo consolidado das métricas"""
        if not self.profile_data:
            return {}

        summary = {"total_steps": len({entry["step"] for entry in self.profile_data}), "total_executions": len(self.profile_data), "total_time": sum(entry["time"] for entry in self.profile_data), "steps": {}}

        for entry in self.profile_data:
            step_name = entry["step"]
            if step_name not in summary["steps"]:
                summary["steps"][step_name] = {"count": 0, "total_time": 0.0, "avg_time": 0.0, "avg_input_len": 0.0, "avg_output_len": 0.0}

            step_stats = summary["steps"][step_name]
            step_stats["count"] += 1
            step_stats["total_time"] += entry["time"]
            step_stats["avg_input_len"] += entry["input_length"]
            step_stats["avg_output_len"] += entry["output_length"]

        for step in summary["steps"].values():
            step["avg_time"] = step["total_time"] / step["count"]
            step["avg_input_len"] /= step["count"]
            step["avg_output_len"] /= step["count"]

        return summary
