from setuptools import setup, find_packages
import pathlib

path = pathlib.Path(__file__).parent
readme = path.joinpath('README.md').read_text()

__version__ = '0.1.1'
setup(
    name='chickadee-opt',
    description='A tool for advanced dispatch optimization of hybrid energy systems',
    long_description_content_type='text/markdown',
    long_description=readme,
    author='Daniel Hill',
    author_email='dhill2522@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[],
    version=__version__,
    url='https://github.com/dhill2522/chickadee',
    python_requires='>=3.5' # because anything less would be really, really sad
)
