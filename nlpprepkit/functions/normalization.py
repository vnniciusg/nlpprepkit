import unicodedata


def normalize_unicode(text: str) -> str:
    """
    normalize unicode characters in the text.

    converts text to its ASCII representation, ignoring non-ASCII characters.

    Args:
        text (str): The input text with unicode characters.

    Returns:
        str: The text with unicode characters normalized.
    """
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")
