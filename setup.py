from setuptools import setup, find_packages

__version__ = '0.0.1'
setup(
    name='chickadee',
    description='chickadee',
    author='Daniel Hill',
    author_email='dhill2522@tutanota.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[],
    version=__version__,
    url='https://github.com/dhill2522/chickadee',
    python_requires='>=3.5' # because anything less would be really, really sad
)
