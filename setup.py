from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()


DESCRIPTION = 'An approach for semi-parametric causal mediation analysis to estimate the natural (in)direct effects of a binary treatment on an outcome of interest.'
LONG_DESCRIPTION = 'A package that uses semi-parametric causal mediation analysis to estimate the natural (in)direct effects of a binary treatment on an outcome of interest.'

# Setting up
setup(
    name="deepmed",
    version='0.2.1',
    license='MIT',
    author="Ge Gao",
    author_email="<gg2797@columbia.edu>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    url='https://github.com/Gloriagao0624/DeepMed_Python',
    keywords=['Causal mediation analysis', 'Semiparametric causal inference', 'Deep neural networks'],
    packages=find_packages(),
    install_requires=['multiprocess'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ]
)