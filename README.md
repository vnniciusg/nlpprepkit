# nlpprepkit

`nlpprepkit` is a Python library for text preprocessing, designed to simplify and accelerate the preparation of text data for natural language processing (NLP) tasks.

## Features

- **Text Cleaning**: Remove extra whitespace, special characters, emojis, HTML tags, URLs, numbers, and social tags.
- **Contraction Expansion**: Expand common English contractions (e.g., "don't" → "do not").
- **Unicode Normalization**: Normalize text to ASCII representation.
- **Pipeline Support**: Create customizable pipelines for sequential text processing.
- **Profiling**: Measure the execution time of each step in the pipeline.
- **Caching**: Avoid redundant processing with built-in caching.
- **Parallel Processing**: Process large text datasets efficiently.

## Installation

Install the library using pip:

```bash
pip install nlpprepkit
```

Or install from source:

```bash
git clone https://github.com/vnniciusg/nlpprepkit.git
cd nlpprepkit
pip install -e .
```

## Quick Start

### Using the Pipeline

```python
from nlpprepkit.pipeline import Pipeline

# Define a custom processing step
def lowercase(text):
    return text.lower()

# Create a pipeline and add the step
pipeline = Pipeline()
pipeline.add_step(lowercase)

# Process text
result = pipeline.process("This is a TEST.")
print(result)  # Output: "this is a test."
```

### Text Cleaning Functions

```python
from nlpprepkit.functions import remove_extra_whitespace, remove_special_characters

text = "This   is   a   test!!!"
cleaned_text = remove_extra_whitespace(text)
print(cleaned_text)  # Output: "This is a test!!!"

cleaned_text = remove_special_characters(cleaned_text)
print(cleaned_text)  # Output: "This is a test"
```

### Expanding Contractions

```python
from nlpprepkit.functions import expand_contractions

text = "I'm going to the store."
expanded_text = expand_contractions(text)
print(expanded_text)  # Output: "I am going to the store."
```

## Running Tests

To run the tests, use `pytest`:

```bash
pytest
```

## Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue on GitHub.

## License

This project is licensed under the [MIT License](LICENSE).
