"""This module provides built-in functions for text cleaning."""

import re


def remove_extra_whitespace(text: str) -> str:
    """
    Removes extra whitespace from the input text.

    Args:
        text (str): The input text with extra whitespace.

    Returns:
        str: The text with extra whitespace removed.
    """
    return re.sub(r"\s+", " ", text).strip()


def remove_special_characters(text: str, keep: str = "") -> str:
    """
    Removes special characters from the input text.

    Args:
        text (str): The input text with special characters.
        keep (str): A string of characters to keep in the text.

    Returns:
        str: The text with special characters removed.
    """
    pattern = f"[^a-zA-Z0-9{re.escape(keep)}]"
    return remove_extra_whitespace(re.sub(pattern, " ", text))


def remove_newline_characters(text: str) -> str:
    """
    Removes newline characters from the input text.

    Args:
        text (str): The input text with newline characters.

    Returns:
        str: The text with newline characters removed.
    """
    return remove_extra_whitespace(re.sub(r"\n", " ", text))


def remove_numbers(text: str) -> str:
    """
    Removes numbers from the input text.

    Args:
        text (str): The input text with numbers.

    Returns:
        str: The text with numbers removed.
    """
    return remove_extra_whitespace(re.sub(r"\d+", "", text))


def remove_urls(text: str) -> str:
    """
    Removes URLs from the input text.

    Args:
        text (str): The input text with URLs.

    Returns:
        str: The text with URLs removed.
    """
    return remove_extra_whitespace(re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE))


def remove_emojis(text: str) -> str:
    """
    Removes emojis from the input text.

    Args:
        text (str): The input text with emojis.

    Returns:
        str: The text with emojis removed.
    """
    emoji_pattern = re.compile("[\U0001f600-\U0001f64f]|[\U0001f300-\U0001f5ff]|[\U0001f680-\U0001f6ff]|[\U0001f1e0-\U0001f1ff]")
    return remove_extra_whitespace(emoji_pattern.sub(r"", text))


def remove_html_tags(text: str) -> str:
    """
    Removes HTML tags from the input text.

    Args:
        text (str): The input text with HTML tags.

    Returns:
        str: The text with HTML tags removed.
    """
    return remove_extra_whitespace(re.sub(r"<.*?>", "", text))


def remove_social_tags(text: str) -> str:
    """
    Removes mentions social tags (e.g., @username, #hashtag) from the input text.

    Args:
        text (str): The input text with mentions.

    Returns:
        str: The text with mentions removed.
    """
    return remove_extra_whitespace(re.sub(r"[@#]\w+", "", text))
