# Chickadee
> A developmental hybrid energy system dispatch optimization package aimed at providing flexibility and development speed while supporting integration with other tools.

The goal of this project is to be a testing-grounds for design and dispatch optimization. This project is meant to stay light, fast and flexible rather than full-featured or rigorous. A key requirement of this project is to maintain compatibility with [HERON](https://github.com/idaholab/HERON) so as to be useable as an internal dispatch optimizer. As a result, the structure necessarily mimics that of HERON to some extent.

## Development Installation
Install chickadee for development using pip. Run the following command from within the root of the repository: `pip install -e .`

## Requirements
- Needs to be able to be imported into HERON for internal dispatch optimization
- Needs to be a flexible place for quickly and simply messing around with dispatch/design opt
- Be importable and useable as a python library
- Has to be useable with various optimizers and with various optimization schemes
- Has to be able to support the same type of components as HERON


## Ideas to experiment with
- Adding DAEs
- Integrating GEKKO as an optimizer
- Support for timing, allowing optimization of algorithm