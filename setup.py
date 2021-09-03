from setuptools import setup
import os
import sys

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name = 'MLAC',
  packages = ['MLAC'],
  version = '0.1',
  license='MIT',
  description = 'Code for comparing different machine learning algorithms for binary classification.',
  long_description_content_type='text/markdown',
  long_description = long_description,
  author = 'Joel Hampton',
  author_email = 'joelelihampton@outlook.com',
  url = 'https://github.com/Joel-H-dot/MLAC',
  keywords = ['Machine Learning', 'Binary Classification', 'SciKit-learn','neural networks'],
  install_requires=[
          'numpy',
          'sklearn',
          'scipy',
          'openpyxl',
          'tensorflow',
            'keras-tuner',
        'scikit-learn',
        'pandas'

      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research ',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
  ],
)