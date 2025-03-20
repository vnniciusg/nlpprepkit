from dataclasses import dataclass, field
from typing import List
import logging

from .exceptions import ConfigurationError


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
    remove_mentions: bool = True
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
    nltk_resources: List[str] = field(
        default_factory=lambda: ["punkt", "wordnet", "stopwords"]
    )
    log_level: int = logging.INFO

    _SUPPORTED_LANGUAGES = {
        "english",
    }

    def __post_init__(self):
        """validate the configuration after initialization."""

        errors: List[str] = []

        # check if all boolean fields are actually boolean
        bool_fields = [
            "expand_contractions",
            "lowercase",
            "remove_urls",
            "remove_newlines",
            "remove_numbers",
            "remove_punctuation",
            "remove_emojis",
            "tokenize",
            "remove_stopwords",
            "stemming",
            "lemmatization",
            "normalize_unicode",
            "remove_mentions",
        ]
        for field_name in bool_fields:
            value = getattr(self, field_name)
            if not isinstance(value, bool):
                errors.append(
                    f"{field_name} must be a boolean value, got {type(value).__name__}"
                )

        # check if language is a string and supported
        if not isinstance(self.language, str):
            errors.append("language must be a string value.")
        elif self.language not in self._SUPPORTED_LANGUAGES:
            errors.append(f"language '{self.language}' is not supported.")

        # check if custom_stopwords, keep_words, and nltk_resources are lists
        list_fields = ["custom_stopwords", "keep_words", "nltk_resources"]
        for field_name in list_fields:
            value = getattr(self, field_name)
            if not isinstance(value, list):
                errors.append(
                    f"{field_name} must be a list, got {type(value).__name__}"
                )

        # check if custom_stopwords and keep_words are lists of strings
        for word in self.custom_stopwords:
            if not isinstance(word, str):
                errors.append(
                    f"custom_stopwords must be a list of strings, found {type(word).__name__}"
                )

        for word in self.keep_words:
            if not isinstance(word, str):
                errors.append(
                    f"keep_words must be a list of strings, found {type(word).__name__}"
                )

        # check if min_word_length and max_word_length are integers and valid
        if not isinstance(self.min_word_length, int):
            errors.append(
                f"min_word_length must be an integer, got {type(self.min_word_length).__name__}"
            )
        elif isinstance(self.min_word_length, int) and self.min_word_length < 1:
            errors.append("min_word_length must be greater than 0")

        if not isinstance(self.max_word_length, int):
            errors.append(
                f"max_word_length must be an integer, got {type(self.max_word_length).__name__}"
            )
        elif (
            isinstance(self.max_word_length, int)
            and isinstance(self.min_word_length, int)
            and self.max_word_length < self.min_word_length
        ):
            errors.append("max_word_length must be greater than min_word_length")

        # check if log_level is a valid logging level
        valid_log_levels = {
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        }
        if self.log_level not in valid_log_levels:
            errors.append(
                f"log_level must be one of {valid_log_levels}, got {self.log_level}"
            )

        # check if only one of stemming or lemmatization is enabled
        if self.stemming and self.lemmatization:
            errors.append("only one of stemming or lemmatization can be enabled")

        if errors:
            raise ConfigurationError("\n".join(errors))
