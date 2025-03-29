"""
This module provides a Pipeline class for creating a text processing pipeline.
It allows adding steps to the pipeline, processing text through the steps, and profiling the execution time of each step.
"""

import pickle
from time import perf_counter
from functools import lru_cache
from typing import Callable, List, Dict, Union, Any, overload
from concurrent.futures import ThreadPoolExecutor
from nlpprepkit.core.utils import ProfilingDecorator, setup_logging
from nlpprepkit.core.exceptions import InvalidInputTypeError, ProfilingNotEnabledError, StepNotCallableError


class Pipeline:
    """A class to create a processing pipeline for text data."""

    def __init__(self, log_level: str = "INFO", *, enable_profiling: bool = False) -> None:
        """
        Initialize the pipeline with an empty list of steps.

        Args:
            log_level (str): The logging level. Default is "INFO".
            enable_profiling (bool): Whether to enable profiling. Default is False.
        """
        self.steps: List[Callable[[str], str]] = []
        self.profile_data: List[Dict[str, Any]] = []
        self.log_level = log_level
        self.enable_profiling = enable_profiling
        self.profiler = ProfilingDecorator()

        self.logger = setup_logging(__name__, self.log_level)

    def add_step(self, func: Callable[[str], str], use_cache: bool = False, max_cache_size: int = 1000) -> None:
        """Add a processing step to the pipeline.

        Args:
            func (Callable[[str], str]): A function that takes a string and returns a string.
            use_cache (bool): Whether to use caching for the step.
        Raises:
            StepNotCallableError: If the provided function is not callable.
        """
        if not callable(func):
            raise StepNotCallableError(func)

        if self.enable_profiling:
            func = self.profiler(func)

        if use_cache:
            func = lru_cache(maxsize=max_cache_size)(func)

        self.steps.append(func)

    @overload
    def process(self, text: str) -> str: ...

    @overload
    def process(self, text: List[str]) -> List[str]: ...

    def process(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """Process the input text through all steps in the pipeline.

        Args:
            text (Union[str, List[str]]): The input text to process. Can be a single string or a list of strings.

        Returns:
            Union[str, List[str]]: The processed text after applying all steps in the pipeline.

        Raises:
            InvalidInputTypeError: If the input is neither a string nor a list of strings.
        """

        def process_single(item: str) -> str:
            for step in self.steps:
                start_time = perf_counter()
                item = step(item)
                elapsed = perf_counter() - start_time
                self.logger.info(f"Step {step.__name__} took {elapsed:.4f} seconds. Input: {len(item)} chars → Output: {len(item)} chars")

            return item

        if isinstance(text, str):
            return process_single(text)

        if isinstance(text, list):
            with ThreadPoolExecutor() as executor:
                return list(executor.map(process_single, text))

        raise InvalidInputTypeError(type(text))

    def __call__(self, text: str) -> str:
        """Enable calling the pipeline instance directly."""
        return self.process(text)

    def clear_steps(self) -> None:
        """Reset the pipeline by removing all steps."""
        self.steps.clear()

    def insert_step(self, index: int, func: Callable[[str], str]) -> None:
        """Insert a processing step at a specific position."""
        self.steps.insert(index, func)

    def remove_step(self, index: int) -> None:
        """Remove a processing step by index."""
        del self.steps[index]

    def save(self, filepath: str) -> None:
        """Save the pipeline to a file using pickle."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    def get_profile_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the profiling data.

        Returns:
            dict: A dictionary containing the profiling summary.

        Raises:
            ProfilingNotEnabledError: If profiling is not enabled.
        """
        if not self.enable_profiling:
            raise ProfilingNotEnabledError()

        return self.profiler.get_summary()

    @classmethod
    def load(cls, filepath: str) -> "Pipeline":
        """Load a pipeline from a file using pickle."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
