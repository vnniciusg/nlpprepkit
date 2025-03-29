import nlpprepkit.functions.text_cleaning as F


def test_remove_extra_whitespace():
    assert F.remove_extra_whitespace("This  is   a   test.") == "This is a test."
    assert F.remove_extra_whitespace("   Leading and trailing spaces   ") == "Leading and trailing spaces"
    assert F.remove_extra_whitespace("No extra spaces") == "No extra spaces"
    assert F.remove_extra_whitespace("") == ""
    assert F.remove_extra_whitespace("  ") == ""
    assert F.remove_extra_whitespace("Multiple   spaces   between   words.") == "Multiple spaces between words."


def test_remove_special_characters():
    assert F.remove_special_characters("Hello, World!") == "Hello World"
    assert F.remove_special_characters("Keep this & that") == "Keep this that"
    assert F.remove_special_characters("12345") == "12345"
    assert F.remove_special_characters("Special characters: @#$%^&*()") == "Special characters"
    assert F.remove_special_characters("") == ""
    assert F.remove_special_characters("No special chars") == "No special chars"
    assert F.remove_special_characters("Keep this & that", keep="&") == "Keep this & that"


def test_remove_newline_characters():
    assert F.remove_newline_characters("Hello\nWorld") == "Hello World"
    assert F.remove_newline_characters("Line 1\nLine 2") == "Line 1 Line 2"
    assert F.remove_newline_characters("No newlines") == "No newlines"
    assert F.remove_newline_characters("") == ""
    assert F.remove_newline_characters("Newline\n") == "Newline"
    assert F.remove_newline_characters("\nNewline") == "Newline"
    assert F.remove_newline_characters("\n") == ""
    assert F.remove_newline_characters("Multiple\nnewlines\nhere") == "Multiple newlines here"


def test_remove_numbers():
    assert F.remove_numbers("There are 123 apples") == "There are apples"
    assert F.remove_numbers("No numbers here") == "No numbers here"
    assert F.remove_numbers("1234567890") == ""
    assert F.remove_numbers("") == ""
    assert F.remove_numbers("Numbers: 1, 2, 3") == "Numbers: , ,"
    assert F.remove_numbers("Keep this number: 42") == "Keep this number:"
    assert F.remove_numbers("Numbers: 1, 2, 3") == "Numbers: , ,"


def test_remove_urls():
    assert F.remove_urls("Visit https://example.com for more info") == "Visit for more info"
    assert F.remove_urls("No URLs here") == "No URLs here"
    assert F.remove_urls("Check out http://example.com") == "Check out"
    assert F.remove_urls("") == ""
    assert F.remove_urls("URL: https://example.com/path?query=123") == "URL:"
    assert F.remove_urls("Keep this URL: http://example.com") == "Keep this URL:"
    assert F.remove_urls("Multiple URLs: http://example.com and https://example.org") == "Multiple URLs: and"


def test_remove_emojis():
    assert F.remove_emojis("Hello 😊") == "Hello"
    assert F.remove_emojis("No emojis here") == "No emojis here"
    assert F.remove_emojis("😊😊😊") == ""
    assert F.remove_emojis("") == ""
    assert F.remove_emojis("Keep this emoji: 😊") == "Keep this emoji:"
    assert F.remove_emojis("Multiple emojis: 😊😊") == "Multiple emojis:"
    assert F.remove_emojis("Emojis: 😊, 😂, 😍") == "Emojis: , ,"


def test_remove_html_tags():
    assert F.remove_html_tags("<p>Hello</p>") == "Hello"
    assert F.remove_html_tags("<div>Test</div>") == "Test"
    assert F.remove_html_tags("No HTML here") == "No HTML here"
    assert F.remove_html_tags("") == ""
    assert F.remove_html_tags("<a href='#'>Link</a>") == "Link"
    assert F.remove_html_tags("<span>Text</span>") == "Text"
    assert F.remove_html_tags("<div><p>Nested</p></div>") == "Nested"
    assert F.remove_html_tags("<div><p>Nested</p></div>") == "Nested"


def test_remove_social_tags():
    assert F.remove_social_tags("Hello @user") == "Hello"
    assert F.remove_social_tags("No social tags here") == "No social tags here"
    assert F.remove_social_tags("@user") == ""
    assert F.remove_social_tags("") == ""
    assert F.remove_social_tags("Keep this @user") == "Keep this"
    assert F.remove_social_tags("Multiple tags: @user1 @user2") == "Multiple tags:"
    assert F.remove_social_tags("Social tags: @user1, @user2") == "Social tags: ,"
    assert F.remove_social_tags("Hashtag #example") == "Hashtag"
    assert F.remove_social_tags("No hashtags here") == "No hashtags here"
    assert F.remove_social_tags("#example") == ""
    assert F.remove_social_tags("") == ""
    assert F.remove_social_tags("Keep this #example") == "Keep this"
    assert F.remove_social_tags("Multiple hashtags: #example1 #example2") == "Multiple hashtags:"
    assert F.remove_social_tags("Hashtags: #example1, #example2") == "Hashtags: ,"
