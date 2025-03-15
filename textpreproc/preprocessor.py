"""
TextPreprocessor: A library for cleaning and preprocessing text data.

This module provides a class for cleaning and preprocessing text data. The class provides
methods for various text preprocessing tasks including contraction expansion, lowercasing,
removing URLs and more.
"""

import logging.config
import re
import os
import json
import logging
import unicodedata
from pathlib import Path
from typing import List, Optional, Union, Dict, Callable, overload
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor

import emoji
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

from model import CleaningConfig


class ProcessingStepRegistry:
    """register the steps to process texts."""

    _text_processors: Dict[str, Callable[[str], str]] = {}
    _token_processors: Dict[str, Callable[[List[str]], List[str]]] = {}

    @classmethod
    def register_text_processor(cls, name: str, func: Optional[Callable] = None):
        """
        register a text processing step.

        Args:
            name (str): the name of the processing step.
            func (Optional[Callable]): the function to be registered.
        """

        def wrapper(func):
            cls._text_processors[name] = func
            return func

        if func is None:
            return wrapper

        cls._text_processors[name] = func
        return func

    @classmethod
    def register_token_processor(cls, name: str, func: Optional[Callable] = None):
        """
        register a token processing step.

        Args:
            name (str): the name of the processing step.
            func (Optional[Callable]): the function to be registered.
        """

        def wrapper(func):
            cls._token_processors[name] = func
            return func

        if func is None:
            return wrapper

        cls._token_processors[name] = func
        return func

    @classmethod
    def get_text_processor(cls, name: str) -> Callable[[str], str]:
        """get a registered text processor function."""
        if name not in cls._text_processors:
            raise ValueError(f"text processor '{name}' not found.")

        return cls._text_processors[name]

    @classmethod
    def get_token_processor(cls, name: str) -> Callable[[List[str]], List[str]]:
        """get a registered token processor function."""
        if name not in cls._token_processors:
            raise ValueError(f"token processor '{name}' not found.")

        return cls._token_processors

    @classmethod
    def list_text_processors(cls) -> List[str]:
        """list all registered text processors."""
        return list(cls._text_processors.keys())

    @classmethod
    def list_token_processors(cls) -> List[str]:
        """list all registered token processors."""
        return list(cls._token_processors.keys())


class TextPreprocessorInterface(ABC):

    @abstractmethod
    def process_text(
        self, text: Union[str, List[str]], max_workers: Optional[int] = None, batch_size: int = 10000
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
                nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
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
            self.pipeline.append(ProcessingStepRegistry.get_text_processor("expand_contractions"))

        if self.config.lowercase:
            self.pipeline.append(ProcessingStepRegistry.get_text_processor("lowercase"))

        if self.config.remove_urls:
            self.pipeline.append(ProcessingStepRegistry.get_text_processor("remove_urls"))

        if self.config.remove_newlines:
            self.pipeline.append(ProcessingStepRegistry.get_text_processor("remove_newlines"))

        if self.config.remove_numbers:
            self.pipeline.append(ProcessingStepRegistry.get_text_processor("remove_numbers"))

        if self.config.normalize_unicode:
            self.pipeline.append(ProcessingStepRegistry.get_text_processor("normalize_unicode"))

        if self.config.remove_punctuation:
            self.pipeline.append(ProcessingStepRegistry.get_text_processor("remove_punctuation"))

        if self.config.remove_emojis:
            self.pipeline.append(ProcessingStepRegistry.get_text_processor("remove_emojis"))

        self.token_pipeline = []

        if self.config.remove_stopwords and "stopwords" in self.config.nltk_resources:
            self.token_pipeline.append(ProcessingStepRegistry.get_token_processor("remove_stopwords"))

        if self.config.stemming:
            self.token_pipeline.append(ProcessingStepRegistry.get_token_processor("stemming"))

        if self.config.lemmatization and "wordnet" in self.config.nltk_resources:
            self.token_pipeline.append(ProcessingStepRegistry.get_token_processor("lemmatization"))

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
    def process_text(self, text: str, max_workers: Optional[int] = None, batch_size: int = 10000) -> str: ...

    @overload
    def process_text(self, text: List[str], max_workers: Optional[int] = None, batch_size: int = 10000) -> List[str]: ...

    def process_text(
        self, text: Union[str, List[str]], max_workers: Optional[int] = None, batch_size: int = 10000
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
            raise ValueError("empty text provided.")

        if max_workers is None:
            max_workers = min(32, os.cpu_count() + 4)

        if not isinstance(text, (str, list)):
            raise ValueError("text must be a string or a list of strings.")

        if isinstance(text, str):
            self.logger.warning("only one text provided. processing sequentially.")
            return self._clean_text(text)

        if not all(isinstance(t, str) for t in text):
            raise ValueError("all elements in the list must be strings.")

        results = []
        for i in range(0, len(text), batch_size):
            batch = text[i : i + batch_size]

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results.extend(list(executor.map(self._clean_text, batch)))

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
                self.logger.error(f"failed to convert text to string: {e}")
                return ""

        if not text:
            self.logger.warning("empty text provided.")
            return ""

        for func in self.pipeline:
            text = func(text)

        if self.config.tokenize:
            tokens = word_tokenize(text)

            if self.config.min_word_length > 1:
                tokens = [token for token in tokens if len(token) >= self.config.min_word_length]

            for func in self.token_pipeline:
                tokens = func(tokens)

            text = " ".join(tokens)

        return text

    @ProcessingStepRegistry.register_text_processor("expand_contractions")
    def _expand_contractions(self, text: str) -> str:
        """expand contractions in the text."""
        return contractions.fix(text)

    @ProcessingStepRegistry.register_text_processor("lowercase")
    def _lowercase(self, text: str) -> str:
        """convert text to lowercase."""
        return text.lower()

    @ProcessingStepRegistry.register_text_processor("remove_urls")
    def _remove_urls(self, text: str) -> str:
        """remove URLs from the text"""
        return re.sub(r"https?://\S+|www\.\S+", "", text)

    @ProcessingStepRegistry.register_text_processor("remove_newlines")
    def _remove_newlines(self, text: str) -> str:
        """remove newlines from the text."""
        return re.sub(r"\n", "", text)

    @ProcessingStepRegistry.register_text_processor("remove_numbers")
    def _remove_numbers(self, text: str) -> str:
        """remove numbers from the text."""
        return re.sub(r"\d+", "", text)

    @ProcessingStepRegistry.register_text_processor("normalize_unicode")
    def _normalize_unicode(self, text: str) -> str:
        """normalize unicode characters in the text."""
        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")

    @ProcessingStepRegistry.register_text_processor("remove_punctuation")
    def _remove_punctuation(self, text: str) -> str:
        """remove punctuation from the text."""
        return re.sub(r"[^\w\s]", "", text)

    @ProcessingStepRegistry.register_text_processor("remove_emojis")
    def _remove_emojis(self, text: str) -> str:
        """remove emojis from the text."""
        return emoji.replace_emoji(text, replace="")

    @ProcessingStepRegistry.register_token_processor("remove_stopwords")
    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """remove stopwords from the tokens."""
        return [token for token in tokens if token not in self.stopwords]

    @ProcessingStepRegistry.register_token_processor("stemming")
    def _stemming(self, tokens: List[str]) -> List[str]:
        """apply stemming to the tokens."""
        return [self.stemmer.stem(token) for token in tokens]

    @ProcessingStepRegistry.register_token_processor("lemmatization")
    def _lemmatization(self, tokens: List[str]) -> List[str]:
        """apply lemmatization to the tokens."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def save_config(self, file_path: Union[str, Path]) -> None:
        """
        save the configuration to a file.

        Args:
            file_path (Union[str, Path]): the path to save the configuration.
        """

        file_path = Path(file_path)

        config_dict = {k: v if not isinstance(v, list) else list(v) for k, v in self.config.__dict__.items()}

        with file_path.open("w") as file:
            json.dump(config_dict, file, indent=4)

        self.logger.info(f"configuration saved to {file_path}")

    @classmethod
    def from_config_file(cls, file_path: Union[str, Path]) -> "TextPreprocessor":
        """
        create a TextPreprocessor instance from a configuration file.

        Args:
            file_path (Union[str, Path]): the path to the configuration file.

        Returns:
            TextPreprocessor: an instance of TextPreprocessor class.
        """

        with open(file_path, "r") as file:
            config_dict = json.load(file)

        config = CleaningConfig(**config_dict)
        return cls(config)

    def _setup_logging(self):
        """setup logging for the class."""
        logging.config.dictConfig(
            {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {"standard": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}},
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
