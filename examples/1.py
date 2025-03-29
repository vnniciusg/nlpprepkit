from nlpprepkit.pipeline import Pipeline

if __name__ == "__main__":

    def lower(text):
        return text.lower()

    pipeline = Pipeline()
    pipeline.add_step(lower)

    print(pipeline.process("This is a test."))
