import sys
from setuptools import setup


if sys.version_info < (3, 0):
    raise ValueError('This package requires python >= 3.0')

with open('prob140/version.py') as fid:
    for line in fid:
        if line.startswith('__version__'):
            version = line.strip().split()[-1][1:-1]
            break


setup(
    name = 'prob140',
    packages = ['prob140'],
    version = version,
    install_requires = [
        'datascience',
        'folium==0.1.5',
        'sphinx',
        'setuptools'
    ],
    description = 'A probability library for Berkeley\'s Prob140 course',
    author = 'Jason Zhang, Dibya Ghosh',
    author_email = 'zhang.j@berkeley.edu',
    url = 'http://prob140.org/',
    license = 'GPL',
    download_url = 'https://gitlab.com/probability/prob140/raw/master/dist/prob140-%s.tar.gz' % version,
    keywords = ['data', 'probability', 'berkeley', 'Prob140'],
    classifiers = []
)