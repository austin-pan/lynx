"""Package definition."""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='lynx',
    version='0.0.1',
    author='Austin Pan',
    # author_email='austinpan8@gmail.com',
    description='Framework for working with tabular data.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url='https://github.com/amirsh/indb-fm/tree/master/polyrecsys',
    license='MIT',
    packages=['lynx'],
    install_requires=['numpy', 'scipy', 'scikit-learn', 'pandas'],
)
