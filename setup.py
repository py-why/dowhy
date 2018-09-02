"""A setuptools based setup module for dowhy.

Adapted from:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='dowhy',

    version='0.1.0',

    description='A Python library for causal inference',  # Required

    long_description=long_description,

    url='https://causalinference.gitlab.io/dowhy',  # Optional

    author='Amit Sharma, Emre Kiciman',


    classifiers=[  # Optional
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='causality causal-inference statistics graphical-model',

    packages=find_packages(exclude=['docs', 'tests']),
    python_requires='>=3.0',

    install_requires=['numpy', 'scikit-learn', 'matplotlib', 'scipy',
                      'pandas', 'networkx', 'sympy'],

)
