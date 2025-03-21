"""
TextCleaner: A comprehensive NLP text preprocessing library.

This library provides tools for cleaning and preprocessing text data
for natural language processing tasks.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("nlpprepkit")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from .preprocessor import TextPreprocessor
from .model import CleaningConfig
from .exceptions import *


def __getattr__(name):
    if name == "functions":
        from . import functions

        return functions
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TextPreprocessor", "CleaningConfig"]

from .functions import (
    expand_contractions,
    lowercase,
    remove_urls,
    remove_newlines,
    remove_numbers,
    normalize_unicode,
    remove_punctuation,
    remove_emojis,
    remove_stopwords,
    stemming,
    lemmatization,
    remove_mentions,
)
