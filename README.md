# nlpprepkit

This can't be the best library for text preprocessing, but it's definitely a library!

## Installation

```bash
pip install nlpprepkit
```

Or install from source:

```bash
git clone https://github.com/vnniciusg/nlpprepkit.git
cd nlpprepkit
pip install -e .
```

## Features

- **Flexible cleaning options**: Control which cleaning operations to apply
- **Parallel processing**: Process large text collections efficiently with multi-core support
- **Caching**: Avoid redundant processing with built-in caching system
- **NLTK integration**: Easy access to stemming, lemmatization, and stopword removal
- **Configurable**: Save and load configuration settings for reproducible workflows

## Quick Start

```python
from nlpprepkit import TextPreprocessor

# Create a preprocessor with default settings
preprocessor = TextPreprocessor()

# Process a single text
cleaned_text = preprocessor.process_text("Check out this URL: https://example.com and these numbers 12345!")
print(cleaned_text) # Output: "check url number"

# Process multiple texts in parallel
texts = [
    "First text with URL https://example.org",
    "Second text with numbers 12345",
    "Third text with emoji 😀 and contraction don't"
]
results = preprocessor.process_text(texts)
```

## Customizing Configuration

```python
from nlpprepkit import TextPreprocessor, CleaningConfig

# Create a custom configuration
config = CleaningConfig(
    expand_contractions=True,
    lowercase=True,
    remove_urls=True,
    remove_newlines=True,
    remove_numbers=True,
    remove_punctuation=True,
    remove_emojis=True,
    tokenize=True,
    remove_stopwords=True,
    stemming=False,
    lemmatization=True, # Only one of stemming or lemmatization can be enabled
    normalize_unicode=True,
    language="english",
    custom_stopwords=["custom1", "custom2"],
    keep_words=["important1", "important2"],
    min_word_length=2,
    max_word_length=15
)

# Create preprocessor with custom config
preprocessor = TextPreprocessor(config)
```

## Saving and Loading Configuration

```python
# Save configuration to a file
preprocessor.save_config("my_config.json")

# Load configuration from a file
preprocessor = TextPreprocessor.from_config_file("my_config.json")
```

## Caching

The library includes a caching system to avoid redundant processing:

```python
# Enable caching (enabled by default)
TextPreprocessor.enable_cache(max_size=1000)

# Clear cache if needed
TextPreprocessor.clear_cache()
```

## Parallel Processing

```python
# Process a large list of texts with parallel processing
results = preprocessor.process_text(
    large_text_list,
    max_workers=8, # Number of parallel workers
    batch_size=1000 # Batch size for processing
)
```

## Available Cleaning Operations

- **Expand contractions**: Convert contractions like "don't" to "do not"
- **Lowercase**: Convert text to lowercase
- **Remove URLs**: Remove web links from text
- **Remove newlines**: Replace newline characters with spaces
- **Remove numbers**: Remove digits from text
- **Remove punctuation**: Remove punctuation marks
- **Remove emojis**: Remove emoji characters
- **Tokenization**: Split text into tokens
- **Remove stopwords**: Remove common words like "the", "a", "is"
- **Stemming/Lemmatization**: Reduce words to their root forms
- **Unicode normalization**: Normalize accented characters
- **Word length filtering**: Filter words by length

## Supported Languages

- English

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)
