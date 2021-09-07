from setuptools import setup

setup(
    name="FANATIC",
    version="1.0",
    author="Ari Silburt",
    author_email="asilburt@bloomberg.net",
    description="FAst Noise-Aware TopIc Clustering",
    url="",
    packages=["fanatic", "fanatic.preprocess", "fanatic.clustering"],
    install_requires=[
        'python_version>="3.7"',
        "gensim==4.0",
        "nltk",
        "numpy==1.20.0",
        "scipy",
        "sklearn",
        "zstandard",
    ],
    zip_safe=False,
)
