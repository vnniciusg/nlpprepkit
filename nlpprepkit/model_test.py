import pytest
import logging

from .model import CleaningConfig
from .exceptions import ConfigurationError


def test_default_config():
    """Test that default configuration initializes without errors."""
    config = CleaningConfig()
    assert config.expand_contractions is True
    assert config.lowercase is True
    assert config.remove_urls is True
    assert config.remove_newlines is True
    assert config.remove_numbers is True
    assert config.remove_punctuation is True
    assert config.remove_emojis is True
    assert config.tokenize is True
    assert config.remove_stopwords is True
    assert config.stemming is False
    assert config.lemmatization is True
    assert config.normalize_unicode is True
    assert config.language == "english"
    assert config.custom_stopwords == []
    assert config.keep_words == []
    assert config.min_word_length == 2
    assert config.max_word_length == 15
    assert config.nltk_resources == ["punkt", "wordnet", "stopwords"]
    assert config.log_level == logging.INFO


def test_custom_config():
    """Test custom configuration settings."""
    config = CleaningConfig(
        expand_contractions=False,
        lowercase=False,
        remove_urls=False,
        custom_stopwords=["custom1", "custom2"],
        keep_words=["keep1", "keep2"],
        min_word_length=3,
        max_word_length=20,
        nltk_resources=["punkt"],
        log_level=logging.DEBUG,
    )

    assert config.expand_contractions is False
    assert config.lowercase is False
    assert config.remove_urls is False
    assert config.custom_stopwords == ["custom1", "custom2"]
    assert config.keep_words == ["keep1", "keep2"]
    assert config.min_word_length == 3
    assert config.max_word_length == 20
    assert config.nltk_resources == ["punkt"]
    assert config.log_level == logging.DEBUG


def test_invalid_boolean_values():
    """Test validation of boolean fields."""
    with pytest.raises(ConfigurationError) as excinfo:
        CleaningConfig(expand_contractions="not a boolean")
    assert "expand_contractions must be a boolean" in str(excinfo.value)


def test_invalid_language():
    """Test validation of language field."""
    with pytest.raises(ConfigurationError) as excinfo:
        CleaningConfig(language="not_supported_language")
    assert "language 'not_supported_language' is not supported" in str(excinfo.value)

    with pytest.raises(ConfigurationError) as excinfo:
        CleaningConfig(language=123)
    assert "language must be a string value" in str(excinfo.value)


def test_invalid_list_fields():
    """Test validation of list fields."""
    with pytest.raises(ConfigurationError) as excinfo:
        CleaningConfig(custom_stopwords="not a list")
    assert "custom_stopwords must be a list" in str(excinfo.value)

    with pytest.raises(ConfigurationError) as excinfo:
        CleaningConfig(keep_words="not a list")
    assert "keep_words must be a list" in str(excinfo.value)

    with pytest.raises(ConfigurationError) as excinfo:
        CleaningConfig(nltk_resources="not a list")
    assert "nltk_resources must be a list" in str(excinfo.value)


def test_invalid_list_content():
    """Test validation of list content."""
    with pytest.raises(ConfigurationError) as excinfo:
        CleaningConfig(custom_stopwords=["valid", 123])
    assert "custom_stopwords must be a list of strings" in str(excinfo.value)

    with pytest.raises(ConfigurationError) as excinfo:
        CleaningConfig(keep_words=["valid", 123])
    assert "keep_words must be a list of strings" in str(excinfo.value)


def test_invalid_word_length():
    """Test validation of word length settings."""
    with pytest.raises(ConfigurationError) as excinfo:
        CleaningConfig(min_word_length="not an int")
    assert "min_word_length must be an integer" in str(excinfo.value)

    with pytest.raises(ConfigurationError) as excinfo:
        CleaningConfig(max_word_length="not an int")
    assert "max_word_length must be an integer" in str(excinfo.value)

    with pytest.raises(ConfigurationError) as excinfo:
        CleaningConfig(min_word_length=0)
    assert "min_word_length must be greater than 0" in str(excinfo.value)

    with pytest.raises(ConfigurationError) as excinfo:
        CleaningConfig(min_word_length=10, max_word_length=5)
    assert "max_word_length must be greater than min_word_length" in str(excinfo.value)


def test_invalid_log_level():
    """Test validation of log level."""
    with pytest.raises(ConfigurationError) as excinfo:
        CleaningConfig(log_level=999)  # Invalid log level
    assert "log_level must be one of" in str(excinfo.value)


def test_stemming_and_lemmatization():
    """Test that stemming and lemmatization can't both be enabled."""
    with pytest.raises(ConfigurationError) as excinfo:
        CleaningConfig(stemming=True, lemmatization=True)
    assert "only one of stemming or lemmatization can be enabled" in str(excinfo.value)

    config1 = CleaningConfig(stemming=True, lemmatization=False)
    assert config1.stemming is True
    assert config1.lemmatization is False

    config2 = CleaningConfig(stemming=False, lemmatization=True)
    assert config2.stemming is False
    assert config2.lemmatization is True

    config3 = CleaningConfig(stemming=False, lemmatization=False)
    assert config3.stemming is False
    assert config3.lemmatization is False
