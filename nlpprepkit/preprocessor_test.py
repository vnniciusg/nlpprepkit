import os
import json
import pytest
from unittest.mock import patch, MagicMock
import tempfile

from .preprocessor import TextPreprocessor
from .model import CleaningConfig
from .exceptions import (
    InputError,
    TokenizationError,
    SerializationError,
    ParallelProcessingError,
)


@pytest.fixture
def simple_text():
    return "This is a simple test text with some URLs like https://example.com and numbers 12345."


@pytest.fixture
def text_list():
    return [
        "First text with URL https://example.org",
        "Second text with numbers 12345",
        "Third text with emoji 😀 and contraction don't",
    ]


@pytest.fixture
def temp_config_file():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
        config_dict = {
            "expand_contractions": True,
            "lowercase": True,
            "remove_urls": True,
            "remove_newlines": False,
            "remove_numbers": False,
            "remove_punctuation": True,
            "remove_emojis": True,
            "remove_mentions": True,
            "tokenize": True,
            "remove_stopwords": False,
            "stemming": False,
            "lemmatization": True,
            "normalize_unicode": True,
            "language": "english",
            "custom_stopwords": [],
            "keep_words": [],
            "min_word_length": 2,
            "max_word_length": 15,
            "nltk_resources": ["punkt", "wordnet"],
            "log_level": 20,
        }
        temp.write(json.dumps(config_dict).encode("utf-8"))

    yield temp.name
    os.unlink(temp.name)


class TestTextPreprocessor:

    def test_initialization(self):
        """Test initialization of TextPreprocessor with default config."""
        preprocessor = TextPreprocessor()

        # Check default pipeline configuration
        assert len(preprocessor.pipeline) > 0
        assert len(preprocessor.token_pipeline) > 0

    def test_custom_initialization(self):
        """Test initialization with custom config."""
        config = CleaningConfig(
            expand_contractions=False,
            lowercase=True,
            remove_urls=False,
            remove_stopwords=False,
            stemming=True,
            lemmatization=False,
        )

        preprocessor = TextPreprocessor(config)

        # Check pipeline configuration based on custom settings
        assert len(preprocessor.pipeline) > 0
        assert hasattr(preprocessor, "stemmer")
        assert not hasattr(preprocessor, "lemmatizer")

    def test_process_single_text(self, simple_text):
        """Test processing a single text."""
        preprocessor = TextPreprocessor()
        result = preprocessor.process_text(simple_text)

        # Verify processing effects
        assert result != simple_text
        assert "https://example.com" not in result
        assert "12345" not in result
        assert result.islower()

    def test_process_text_list(self, text_list):
        """Test processing a list of texts."""
        preprocessor = TextPreprocessor()
        results = preprocessor.process_text(text_list)

        # Verify results type and length
        assert isinstance(results, list)
        assert len(results) == len(text_list)

        # Verify each text was processed
        for i, result in enumerate(results):
            assert result != text_list[i]
            assert result.islower()

    def test_empty_input(self):
        """Test handling empty input."""
        preprocessor = TextPreprocessor()

        with pytest.raises(InputError):
            preprocessor.process_text("")

        with pytest.raises(InputError):
            preprocessor.process_text([])

    def test_invalid_input_type(self):
        """Test handling invalid input types."""
        preprocessor = TextPreprocessor()

        with pytest.raises(InputError):
            preprocessor.process_text(123)

        with pytest.raises(ValueError):
            preprocessor.process_text(["valid", 123, "also valid"])

    @patch("nlpprepkit.preprocessor.word_tokenize")
    def test_tokenization_error(self, mock_tokenize, simple_text):
        """Test handling tokenization errors."""
        mock_tokenize.side_effect = Exception("Tokenization failed")

        preprocessor = TextPreprocessor()
        preprocessor.config.tokenize = True
        preprocessor.clear_cache()

        with pytest.raises(TokenizationError):
            preprocessor.process_text(simple_text)

    @patch("nlpprepkit.preprocessor.ProcessPoolExecutor")
    def test_parallel_processing_error(self, mock_executor, text_list):
        """Test handling parallel processing errors."""
        mock_executor_instance = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        mock_executor_instance.map.side_effect = Exception("Processing failed")

        preprocessor = TextPreprocessor()

        with pytest.raises(ParallelProcessingError):
            preprocessor.process_text(text_list)

    def test_save_config(self, temp_config_file):
        """Test saving configuration to file."""
        config = CleaningConfig(expand_contractions=False, lowercase=False)

        preprocessor = TextPreprocessor(config)
        new_config_path = f"{temp_config_file}_new"

        preprocessor.save_config(new_config_path)

        assert os.path.exists(new_config_path)

        with open(new_config_path, "r") as f:
            saved_config = json.load(f)
            assert saved_config["expand_contractions"] is False
            assert saved_config["lowercase"] is False

        os.unlink(new_config_path)

    def test_from_config_file(self, temp_config_file):
        """Test loading configuration from file."""
        preprocessor = TextPreprocessor.from_config_file(temp_config_file)

        assert preprocessor.config.expand_contractions is True
        assert preprocessor.config.lowercase is True
        assert preprocessor.config.remove_urls is True
        assert preprocessor.config.remove_newlines is False
        assert preprocessor.config.remove_numbers is False

    def test_invalid_config_file(self):
        """Test handling invalid configuration file."""
        with pytest.raises(SerializationError):
            TextPreprocessor.from_config_file("nonexistent_file.json")


def test_stopwords_removal():
    """Test stopwords removal functionality."""
    config = CleaningConfig(
        expand_contractions=False,
        lowercase=True,
        remove_urls=False,
        remove_emojis=False,
        remove_numbers=False,
        remove_punctuation=False,
        remove_stopwords=True,
        stemming=False,
        lemmatization=False,
    )

    preprocessor = TextPreprocessor(config)

    # Process text with common stopwords
    result = preprocessor.process_text("The quick brown fox jumps over the lazy dog")

    # Verify stopwords were removed
    assert "the" not in result.split()
    assert "over" not in result.split()
    assert "quick" in result.split()
    assert "brown" in result.split()
    assert "fox" in result.split()


def test_stemming():
    """Test stemming functionality."""
    config = CleaningConfig(
        expand_contractions=False,
        lowercase=True,
        remove_urls=False,
        remove_emojis=False,
        remove_numbers=False,
        remove_punctuation=False,
        remove_stopwords=False,
        stemming=True,
        lemmatization=False,
    )

    preprocessor = TextPreprocessor(config)

    # Process text with words that can be stemmed
    result = preprocessor.process_text("Running jumps flies coding")

    # Verify stemming was applied
    words = result.split()
    assert "run" in words
    assert "jump" in words
    assert "fli" in words
    assert "code" in words


def test_lemmatization():
    """Test lemmatization functionality."""
    config = CleaningConfig(
        expand_contractions=False,
        lowercase=True,
        remove_urls=False,
        remove_emojis=False,
        remove_numbers=False,
        remove_punctuation=False,
        remove_stopwords=False,
        stemming=False,
        lemmatization=True,
    )

    preprocessor = TextPreprocessor(config)

    # Process text with words that can be lemmatized
    result = preprocessor.process_text("better studies running flies")

    # Verify lemmatization was applied
    words = result.split()
    assert "better" in words
    assert "study" in words
    assert "running" in words
    assert "fly" in words


def test_min_word_length_filtering():
    """Test minimum word length filtering."""
    config = CleaningConfig(
        min_word_length=4,
        tokenize=True,
        expand_contractions=False,
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=False,
        stemming=False,
        lemmatization=False,
    )

    preprocessor = TextPreprocessor(config)

    # Process text with words of different lengths
    result = preprocessor.process_text("The big fox ate the small rat")

    # Verify short words were removed
    words = result.split()
    assert "the" not in words
    assert "big" not in words
    assert "fox" not in words
    assert "ate" not in words
    assert "small" in words
    assert "rat" not in words
