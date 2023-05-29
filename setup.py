from setuptools import setup, find_packages

setup(
    name="algovate",
    version="0.0.1",
    description="Automating the evaluation of retrieval QandA",  
    long_description=open('README.md').read(), 
    url="https://github.com/algoveraai/algovate",   
    packages=find_packages(),
    python_requires='>=3.6',
)