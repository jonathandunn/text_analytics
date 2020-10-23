import os
import setuptools
from setuptools import setup, find_packages
from distutils.core import setup

# Utility function to read the README file.

def read(fname):
	return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
	name = "text_analytics",
	version = "1.0",
	author = "Jonathan Dunn",
	author_email = "jonathan.dunn@canterbury.ac.nz",
	description = ("Basic text analytics and natural language processing in Python"),
	license = "GNU GENERAL PUBLIC LICENSE v3",
	url = "https://github.com/jonathandunn/text_analytics",
	keywords = "text analytics, natural language processing, computational linguistics",
	packages = find_packages(exclude=["*.pyc", "__pycache__"]),
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
						],
	include_package_data=True,
	long_description=read('README.md'),
	)