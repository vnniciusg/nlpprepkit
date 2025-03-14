"""
TextPreprocessor: A library for cleaning and preprocessing text data.

This module provides a class for cleaning and preprocessing text data. The class provides
methods for various text preprocessing tasks including contraction expansion, lowercasing,
removing URLs and more.
"""

import logging.config
import re
import json
import logging
import unicodedata
from pathlib import Path
from typing import List, Optional, Union
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor

import spacy
import emoji
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


@dataclass
class CleaningConfig:
    """configuration for text cleaning operations."""

    expand_contractions: bool = True
    lowercase: bool = True
    remove_urls: bool = True
    remove_newlines: bool = True
    remove_numbers: bool = True
    remove_punctuation: bool = True
    remove_emojis: bool = True
    tokenize: bool = True
    remove_stopwords: bool = True
    stemming: bool = False
    lemmatization: bool = True
    normalize_unicode: bool = True
    language: str = "english"
    custom_stopwords: List[str] = field(default_factory=list)
    keep_words: List[str] = field(default_factory=list)
    min_word_length: int = 2
    max_word_length: int = 15
    nltk_resources: List[str] = field(default_factory=lambda: ["punkt", "wordnet", "stopwords"])
    log_level: int = logging.INFO


class TextPreprocessor:
    """
    a class for cleaning and preprocessing text data>

    this class provides methods for various text preprocessing tasks including contraction expansion,
    lowercasing, removing URLs and more.
    """

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

        if self.config.lemmatization and not self.config.tokenize:
            try:
                self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            except OSError:
                self.logger.warning("spacy model not found. using nltk lemmatizer instead.")
                self.nlp = None

    def _configure_pipeline(self):
        """configure the preprocessing pipeline."""

        self.pipeline = []

        if self.config.expand_contractions:
            self.pipeline.append(self._expand_contractions)

        if self.config.lowercase:
            self.pipeline.append(self._lowercase)

        if self.config.remove_urls:
            self.pipeline.append(self._remove_urls)

        if self.config.remove_newlines:
            self.pipeline.append(self._remove_newlines)

        if self.config.remove_numbers:
            self.pipeline.append(self._remove_numbers)

        if self.config.normalize_unicode:
            self.pipeline.append(self._normalize_unicode)

        if self.config.remove_punctuation:
            self.pipeline.append(self._remove_punctuation)

        if self.config.remove_emojis:
            self.pipeline.append(self._remove_emojis)

        self.token_pipeline = []

        if self.config.remove_stopwords and "stopwords" in self.config.nltk_resources:
            self.token_pipeline.append(self._remove_stopwords)

        if self.config.stemming:
            self.token_pipeline.append(self._stemming)

        if self.config.lemmatization and "wordnet" in self.config.nltk_resources:
            self.token_pipeline.append(self._lemmatization)

    def clean_text(self, text: str) -> str:
        """
        clean and preprocess text acccording to the configured pipeline.

        Args:
            text (str): the input text to be cleaned.

        Returns:
            str: the cleaned text.
        """

        if not isinstance(text, str) or not text:
            self.logger.warning("input text is empty or not a string.")
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

    def process_texts(self, texts: List[str], max_workes: Optional[int] = None) -> List[str]:
        """
        process a list of texts in parallel.

        Args:
            texts (List[str]): a list of texts to be processed.
            max_workes (Optional[int]): the number of workers to use for parallel processing.

        Returns:
            List[str]: a list of cleaned texts.
        """
        if len(texts) <= 1:
            self.logger.warning("only one text provided. processing sequentially.")
            return [self.clean_text(text) for text in texts]

        with ProcessPoolExecutor(max_workes) as executor:
            results = list(executor.map(self.clean_text, texts))

        return results

    def _expand_contractions(self, text: str) -> str:
        """expand contractions in the text."""
        return contractions.fix(text)

    def _lowercase(self, text: str) -> str:
        """convert text to lowercase."""
        return text.lower()

    def _remove_urls(self, text: str) -> str:
        """remove URLs from the text"""
        return re.sub(r"https?://\S+|www\.\S+", "", text)

    def _remove_newlines(self, text: str) -> str:
        """remove newlines from the text."""
        return re.sub(r"\n", "", text)

    def _remove_numbers(self, text: str) -> str:
        """remove numbers from the text."""
        return re.sub(r"\d+", "", text)

    def _normalize_unicode(self, text: str) -> str:
        """normalize unicode characters in the text."""
        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")

    def _remove_punctuation(self, text: str) -> str:
        """remove punctuation from the text."""
        return re.sub(r"[^\w\s]", "", text)

    def _remove_emojis(self, text: str) -> str:
        """remove emojis from the text."""
        return emoji.replace_emoji(text, replace="")

    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """remove stopwords from the tokens."""
        return [token for token in tokens if token not in self.stopwords]

    def _stemming(self, tokens: List[str]) -> List[str]:
        """apply stemming to the tokens."""
        return [self.stemmer.stem(token) for token in tokens]

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
