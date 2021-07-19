import os
from setuptools import find_packages
from distutils.core import setup
# Utility function to read the README file.

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="text_analytics",
    version="1.0.1",
    author="Jonathan Dunn",
    author_email="jonathan.dunn@canterbury.ac.nz",
    description="Basic text analytics and natural language processing in Python",
    license="GNU GENERAL PUBLIC LICENSE v3",
    url="https://github.com/jonathandunn/text_analytics",
    keywords="text analytics, natural language processing, computational linguistics",
    packages=find_packages(exclude=["*.pyc", "__pycache__"]),
    package_data={'': []},
    install_requires=["cytoolz",
                      "gensim",
                      "numpy",
                      "pandas",
                      "scipy",
                      "sklearn",
                      "tensorflow",
                      "spacy",
                      "wordcloud",
                      "matplotlib",
                      "python-Levenshtein",
                      "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz",
                      "requests",
                      "dropbox",
                      "stop-words",
                      "boto3",
                      "tqdm"],
    include_package_data=True,
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
)
