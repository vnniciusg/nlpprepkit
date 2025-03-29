from nlpprepkit.functions import expand_contractions


def test_expand_contractions():
    test_cases = [
        ("I'm going to the store.", "I am going to the store."),
        ("He's not here.", "He is not here."),
        ("They've been waiting.", "They have been waiting."),
        ("She'll be there soon.", "She will be there soon."),
        ("We can't do that.", "We cannot do that."),
        ("You don't understand.", "You do not understand."),
        ("It's a nice day.", "It is a nice day."),
        ("I'd like some coffee.", "I would like some coffee."),
        ("Aren't you coming?", "Are not you coming?"),
        ("How'd you do that?", "How did you do that?"),
    ]

    for text, expected in test_cases:
        result = expand_contractions(text)
        assert result == expected, f"Expected: {expected}, but got: {result}"
