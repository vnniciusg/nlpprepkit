"""
TextPreprocessor: A library for cleaning and preprocessing text data.

This module provides a class for cleaning and preprocessing text data. The class provides
methods for various text preprocessing tasks including contraction expansion, lowercasing,
removing URLs and more.
"""

import logging.config
import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, overload
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tqdm import tqdm

from .model import CleaningConfig
from . import functions as F
from .exceptions import (
    SerializationError,
    TokenizationError,
    InputError,
    ParallelProcessingError,
)
from .utils import generate_cache_key


class TextPreprocessorInterface(ABC):

    @abstractmethod
    def process_text(
        self,
        text: Union[str, List[str]],
        max_workers: Optional[int] = None,
        batch_size: int = 10000,
    ) -> Union[str, List[str]]:
        pass

    @abstractmethod
    def save_config(self, file_path: Union[str, Path]) -> None:
        pass

    @classmethod
    @abstractmethod
    def from_config_file(cls, file_path: Union[str, Path]) -> "TextPreprocessor":
        pass


class TextPreprocessor(TextPreprocessorInterface):
    """
    a class for cleaning and preprocessing text data>

    this class provides methods for various text preprocessing tasks including contraction expansion,
    lowercasing, removing URLs and more.
    """

    _cache: Dict[str, str] = {}
    _cache_enabled: bool = True
    _cache_max_size: int = 1000

    def __init__(self, config: Optional[CleaningConfig] = None):
        """
        initialize the preprocessor.

        Args:
            config (Optional[CleaningConfig]): an instance of CleaningConfig class.
        """
        self.config = config if config else CleaningConfig()
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        self._dowload_nltk_resources()
        self._initialize_resources()
        self._configure_pipeline()

    def _dowload_nltk_resources(self):
        """download required nltk resources."""

        for resource in self.config.nltk_resources:
            try:
                nltk.data.find(
                    f"tokenizers/{resource}"
                    if resource == "punkt"
                    else f"corpora/{resource}"
                )
                self.logger.debug(f"NLTK resource '{resource}' already downloaded.")
            except LookupError:
                self.logger.info(f"Downloading NLTK resource '{resource}'...")
                nltk.download(resource)

    def _initialize_resources(self):
        """initialize resources for text preprocessing."""

        if self.config.remove_stopwords:
            self.stopwords = set(stopwords.words(self.config.language))

            if self.config.custom_stopwords:
                self.stopwords.update(self.config.custom_stopwords)

            if self.config.keep_words:
                self.stopwords = self.stopwords.difference(self.config.keep_words)

        if self.config.lemmatization and "wordnet" in self.config.nltk_resources:
            self.lemmatizer = WordNetLemmatizer()

        if self.config.stemming:
            self.stemmer = PorterStemmer()

    def _configure_pipeline(self):
        """configure the preprocessing pipeline."""

        self.pipeline = []

        if self.config.expand_contractions:
            self.pipeline.append(F.expand_contractions)

        if self.config.lowercase:
            self.pipeline.append(F.lowercase)

        if self.config.remove_urls:
            self.pipeline.append(F.remove_urls)

        if self.config.remove_newlines:
            self.pipeline.append(F.remove_newlines)

        if self.config.remove_numbers:
            self.pipeline.append(F.remove_numbers)

        if self.config.normalize_unicode:
            self.pipeline.append(F.normalize_unicode)

        if self.config.remove_punctuation:
            self.pipeline.append(F.remove_punctuation)

        if self.config.remove_emojis:
            self.pipeline.append(F.remove_emojis)

        if self.config.remove_mentions:
            self.pipeline.append(F.remove_mentions)

        self.token_pipeline = []

        if self.config.remove_stopwords and "stopwords" in self.config.nltk_resources:
            self.token_pipeline.append(
                {"func": F.remove_stopwords, "args": {"stopwords": self.stopwords}}
            )

        if self.config.stemming:
            self.token_pipeline.append(
                {"func": F.stemming, "args": {"stemmer": self.stemmer}}
            )

        if self.config.lemmatization and "wordnet" in self.config.nltk_resources:
            self.token_pipeline.append(
                {"func": F.lemmatization, "args": {"lemmatizer": self.lemmatizer}}
            )

    @classmethod
    def enable_cache(cls, enabled: bool = True, max_size: int = 1000):
        """enable or disable caching of processed texts."""
        cls._cache_enabled = enabled
        cls._cache_max_size = max_size
        if enabled:
            cls._cache.clear()

    @classmethod
    def clear_cache(cls):
        """clear the cache of processed texts."""
        cls._cache.clear()

    @overload
    def process_text(
        self, text: str, max_workers: Optional[int] = None, batch_size: int = 10000
    ) -> str: ...

    @overload
    def process_text(
        self,
        text: List[str],
        max_workers: Optional[int] = None,
        batch_size: int = 10000,
    ) -> List[str]: ...

    def process_text(
        self,
        text: Union[str, List[str]],
        max_workers: Optional[int] = None,
        batch_size: int = 10000,
    ) -> Union[str, List[str]]:
        """
        process a list of texts in parallel.

        Args:
            text (Union[str, List[str]]): a single text or a list of texts to be processed.
            max_workers (Optional[int]): the number of workers to use for parallel processing (default: min(32, os.cpu_count() + 4)).
            batch_size (int): the number of texts to process in each batch.
        Returns:
            Union[str, List[str]]: the cleaned text or a list of cleaned texts.
        """

        if not text:
            raise InputError("empty text provided.")
        if not isinstance(text, (str, list)):
            raise InputError("text must be a string or a list of strings.")

        if isinstance(text, str):
            self.logger.warning("only one text provided. processing sequentially.")
            return self._clean_text(text)

        if not all(isinstance(t, str) for t in text):
            raise ValueError("all elements in the list must be strings.")

        self.logger.info(
            f"processing {len(text)} texts with {max_workers or 'default'} workers..."
        )
        try:
            with ProcessPoolExecutor(
                max_workers=max_workers or min(32, os.cpu_count() + 4)
            ) as executor:
                results = list(
                    tqdm(
                        executor.map(self._clean_text, text, chunksize=batch_size),
                        total=len(text),
                        desc="Processing texts",
                    )
                )
        except Exception as e:
            raise ParallelProcessingError("failed to process texts.") from e

        return results

    def _clean_text(self, text: str) -> str:
        """
        clean and preprocess text acccording to the configured pipeline.

        Args:
            text (str): the input text to be cleaned.

        Returns:
            str: the cleaned text.
        """

        if not isinstance(text, str):
            try:
                text = str(text)
                self.logger.warning(f"input text converted to string: {text}")
            except Exception as e:
                raise InputError("failed to convert input text to string.") from e

        if not text:
            self.logger.warning("empty text provided.")
            return ""

        cache_key = generate_cache_key(text, self.config)
        if self._cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        for func in self.pipeline:
            text = func(text)

        if self.config.tokenize:
            try:
                tokens = word_tokenize(text)

                if self.config.min_word_length > 1 and self.config.max_word_length > 1:
                    tokens = [
                        token
                        for token in tokens
                        if len(token) >= self.config.min_word_length
                        and len(token) <= self.config.max_word_length
                    ]

                for pipeline_item in self.token_pipeline:
                    func = pipeline_item["func"]
                    params = pipeline_item["args"]

                    tokens = func(tokens, **params)

                text = " ".join(tokens)
            except Exception as e:
                raise TokenizationError("failed to tokenize text.") from e

        self._cache[cache_key] = text
        if self._cache_enabled and len(self._cache) > self._cache_max_size:
            self._cache.popitem(last=False)

        return text

    def save_config(self, file_path: Union[str, Path]) -> None:
        """
        save the configuration to a file.

        Args:
            file_path (Union[str, Path]): the path to save the configuration.
        """

        file_path = Path(file_path)
        config_dict = {
            k: v if not isinstance(v, list) else list(v)
            for k, v in self.config.__dict__.items()
        }

        try:
            with file_path.open("w") as file:
                json.dump(config_dict, file, indent=4)
            self.logger.info(f"configuration saved to {file_path}")
        except Exception as e:
            self.logger.error(f"failed to save configuration to file: {e}")
            raise SerializationError("failed to save configuration to file.") from e

    @classmethod
    def from_config_file(cls, file_path: Union[str, Path]) -> "TextPreprocessor":
        """
        create a TextPreprocessor instance from a configuration file.

        Args:
            file_path (Union[str, Path]): the path to the configuration file.

        Returns:
            TextPreprocessor: an instance of TextPreprocessor class.
        """

        try:
            with open(file_path, "r") as file:
                config_dict = json.load(file)
            config = CleaningConfig(**config_dict)
            return cls(config)
        except Exception as e:
            raise SerializationError("failed to load configuration from file.") from e

    def _setup_logging(self):
        """setup logging for the class."""
        logging.config.dictConfig(
            {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "standard": {
                        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    }
                },
                "handlers": {
                    "default": {
                        "level": self.config.log_level,
                        "formatter": "standard",
                        "class": "logging.StreamHandler",
                    },
                },
                "loggers": {
                    __name__: {
                        "handlers": ["default"],
                        "level": self.config.log_level,
                        "propagate": False,
                    },
                },
            }
        )
