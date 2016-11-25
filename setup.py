import sys
from setuptools import setup

version = 0.1

if sys.version_info < (3, 0):
    raise ValueError('This package requires python >= 3.0')

with open('requirements.txt') as fid:
    install_requires = [l.strip() for l in fid.readlines() if l]


setup(
    name = 'prob140',
    packages = ['prob140'],
    version = version,
    install_requires = install_requires,
    description = 'A probability library for Berkeley\'s Prob140 course',
    author = 'Jason Zhang, Dibya Ghosh',
    author_email = 'zhang.j@berkeley.edu',
    url = 'prob140.org',
    download_url = 'https://gitlab.com/probability/prob140',
    keywords = ['data', 'probability', 'berkeley', 'Prob140'],
    classifiers = []
)