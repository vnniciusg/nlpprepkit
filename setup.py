from setuptools import setup, find_packages

setup(
    name="textpreprocessor",
    version="0.1.0",
    description="Text Preprocessing Library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="vnniciusg",
    author_email="vnniciusg@gmail.com",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=[
        "contractions>=0.1.73",
        "emoji>=2.14.1",
        "nltk>=3.9.1",
        "tqdm>=4.67.1",
        "unidecode>=1.2.0",
    ],
)