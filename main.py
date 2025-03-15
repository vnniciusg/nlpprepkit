from textpreproc import TextPreprocessor

if __name__ == "__main__":
    text = "Hello, World!"
    tp = TextPreprocessor()
    print(tp.process_text(text))