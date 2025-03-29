from nlpprepkit.functions import normalize_unicode


def test_normalize_unicode():
    assert normalize_unicode("Café") == "Cafe"
    assert normalize_unicode("Résumé") == "Resume"
    assert normalize_unicode("naïve") == "naive"
    assert normalize_unicode("Crème brûlée") == "Creme brulee"
    assert normalize_unicode("Hello, World!") == "Hello, World!"
    assert normalize_unicode("") == ""
    assert normalize_unicode("12345") == "12345"
    assert normalize_unicode("😊") == ""
    assert normalize_unicode("Café Résumé naïve Crème brûlée") == "Cafe Resume naive Creme brulee"
    assert normalize_unicode("Café") == "Cafe"
