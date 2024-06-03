from setuptools import find_packages, setup


setup(
    name='ollama_prompter', 
    version='0.0.1', 
    description='Open-source library of ConFit', 
    author='Yang Wang', 
    author_email='yangwang4work@gmail.com', 
    package_dir={'': 'src'}, 
    packages=find_packages('src'), 
    python_requires='>=3.9.0', 
    keywords='ollama prompt'
)