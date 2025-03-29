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
