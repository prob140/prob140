import sys
from setuptools import setup


if sys.version_info < (3, 0):
    raise ValueError('This package requires python >= 3.0')

VERSION = ''
with open('prob140/version.py') as fid:
    for line in fid:
        if line.startswith('__version__'):
            VERSION = line.strip().split()[-1][1:-1]
            break
assert VERSION != ''

DOWNLOAD_URL = ('https://github.com/prob140/prob140/blob/master/dist/'
                'prob140-{}.tar.gz'.format(VERSION))


setup(
    name='prob140',
    packages=['prob140'],
    version=VERSION,
    install_requires=[
        'datascience',
        'folium',
        'sphinx',
        'setuptools',
        'sympy'
    ],
    description='A probability library for Berkeley\'s Prob140 course',
    author='Jason Zhang, Dibya Ghosh',
    author_email='zhang.j@berkeley.edu',
    url='https://github.com/prob140/prob140',
    license='CC BY-NC-ND 4.0',
    download_url=DOWNLOAD_URL,
    keywords=['data', 'probability', 'berkeley', 'Prob140'],
    classifiers=[
        'Programming Language :: Python :: 3'
    ]
)