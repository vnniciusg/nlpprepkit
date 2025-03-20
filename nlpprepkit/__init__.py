"""
TextCleaner: A comprehensive NLP text preprocessing library.

This library provides tools for cleaning and preprocessing text data
for natural language processing tasks.
"""

__version__ = "1.0.4"

missing_deps = []

try:
    import nltk
except ImportError:
    missing_deps.append("nltk")

try:
    import contractions
except ImportError:
    missing_deps.append("contractions")

try:
    import emoji
except ImportError:
    missing_deps.append("emoji")

if missing_deps:
    deps_str = ", ".join(missing_deps)
    install_cmd = f"pip install {' '.join(missing_deps)}"
    uv_cmd = f"uv add {' '.join(missing_deps)}"

    error_msg = (
        f"Missing required dependencies: {deps_str}\n\n"
        f"Please install them using one of the following commands:\n"
        f"- Using pip: {install_cmd}\n"
        f"- Using uv: {uv_cmd}\n\n"
        f"Or install all dependencies with: pip install -r requirements.txt"
    )

    raise ImportError(error_msg)

from .preprocessor import TextPreprocessor
from .model import CleaningConfig
from .functions import *
from .exceptions import *

__all__ = [
    "TextPreprocessor",
    "CleaningConfig",
    "expand_contractions",
    "lowercase",
    "remove_urls",
    "remove_newlines",
    "remove_numbers",
    "normalize_unicode",
    "remove_punctuation",
    "remove_emojis",
    "remove_stopwords",
    "stemming",
    "lemmatization",
]
