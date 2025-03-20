import re
import unicodedata
from typing import List

import emoji
import contractions


def expand_contractions(text: str) -> str:
    """
    expand contractions in the text.

    Example:
        "don't" -> "do not"
    """
    return contractions.fix(text)


def lowercase(text: str) -> str:
    """
    convert text to lowercase.
    """
    return text.lower()


def remove_urls(text: str) -> str:
    """
    remove URLs from the text.

    Example:
        "Visit https://example.com for details." -> "Visit  for details."
    """
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def remove_newlines(text: str) -> str:
    """
    remove newline characters from the text.
    """
    return re.sub(r"\n", " ", text)


def remove_numbers(text: str) -> str:
    """
    remove numbers from the text.
    """
    return re.sub(r"\d+", "", text)


def normalize_unicode(text: str) -> str:
    """
    normalize unicode characters in the text.

    converts text to its ASCII representation, ignoring non-ASCII characters.
    """
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )


def remove_punctuation(text: str) -> str:
    """
    remove punctuation from the text.
    """
    return re.sub(r"[^\w\s]", "", text)


def remove_emojis(text: str) -> str:
    """
    remove emojis from the text.
    """
    return emoji.replace_emoji(text, replace="")


def remove_stopwords(tokens: List[str], stopwords: set) -> List[str]:
    """
    remove stopwords from a list of tokens.

    Args:
        tokens (List[str]): List of tokens.
        stopwords (set): A set of stopwords to remove.

    Returns:
        List[str]: Filtered tokens without stopwords.
    """
    return [token for token in tokens if token not in stopwords]


def stemming(tokens: List[str], stemmer) -> List[str]:
    """
    apply stemming to a list of tokens.

    Args:
        tokens (List[str]): List of tokens.
        stemmer: An instance of a stemmer (e.g., nltk's PorterStemmer).

    Returns:
        List[str]: Stemmed tokens.
    """
    return [stemmer.stem(token) for token in tokens]


def lemmatization(tokens: List[str], lemmatizer) -> List[str]:
    """
    apply lemmatization to a list of tokens.

    Args:
        tokens (List[str]): List of tokens.
        lemmatizer: An instance of a lemmatizer (e.g., nltk's WordNetLemmatizer).

    Returns:
        List[str]: Lemmatized tokens.
    """
    return [lemmatizer.lemmatize(token) for token in tokens]


def remove_mentions(text: str) -> str:
    """
    remove all the mentions (@username) in the text.

    Args:
        text (str): the orignal text containing the mentions.

    Returns:
        str: the text withou mentions.
    """

    text_cleaned = re.sub(r"@\w+", "", text)
    return " ".join(text_cleaned.split())
