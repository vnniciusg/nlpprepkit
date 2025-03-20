from nltk.stem import PorterStemmer, WordNetLemmatizer
from . import functions as F


def test_expand_contractions():
    assert F.expand_contractions("I'm happy") == "I am happy"
    assert F.expand_contractions("I've been there") == "I have been there"
    assert F.expand_contractions("I don't know") == "I do not know"


def test_lower_case():
    assert F.lowercase("Hello World") == "hello world"
    assert F.lowercase("Hello") == "hello"
    assert F.lowercase("HELLO") == "hello"


def test_remove_urls():
    assert (
        F.remove_urls("Visit https://example.com for details.") == "Visit  for details."
    )
    assert F.remove_urls("Check www.example.com") == "Check "


def test_remove_newlines():
    assert F.remove_newlines("Hello\nWorld") == "Hello World"
    assert F.remove_newlines("Line1\nLine2\nLine3") == "Line1 Line2 Line3"


def test_remove_numbers():
    assert F.remove_numbers("There are 123 apples") == "There are  apples"
    assert F.remove_numbers("2021 is a year") == " is a year"


def test_normalize_unicode():
    assert F.normalize_unicode("café") == "cafe"
    assert F.normalize_unicode("naïve") == "naive"


def test_remove_punctuation():
    assert F.remove_punctuation("Hello, world!") == "Hello world"
    assert F.remove_punctuation("Python's great.") == "Pythons great"


def test_remove_emojis():
    assert F.remove_emojis("Hello 😊") == "Hello "
    assert F.remove_emojis("Good morning 🌞") == "Good morning "


def test_remove_stopwords():
    tokens = ["this", "is", "a", "test"]
    stopwords = {"is", "a"}
    assert F.remove_stopwords(tokens, stopwords) == ["this", "test"]


def test_stemming():
    tokens = ["running", "jumps", "easily"]
    stemmer = PorterStemmer()
    assert F.stemming(tokens, stemmer) == ["run", "jump", "easili"]


def test_lemmatization():
    tokens = ["running", "jumps", "easily"]
    lemmatizer = WordNetLemmatizer()
    assert F.lemmatization(tokens, lemmatizer) == ["running", "jump", "easily"]


def test_remove_mentions():
    text = "Hello @user1, how are you? @user2, did you see that?"
    assert F.remove_mentions(text) == "Hello , how are you? , did you see that?"
