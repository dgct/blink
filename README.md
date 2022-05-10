# BLINK

Blurry linkage (BLINK) is a Python library 
for fast and efficient cosine scoring of sparse data 
without binning. Orginally designed for fragmentation 
mass spectra, BLINK has been abstracted for use with 
arbitrary data types while maintaining capabilities
relevent to mass spec like neutral-losses and mass shifts.

## Use cases:

- Quickly score experimental MS2 against spectral libraries.
(~20 million comparisons per second on 2021 MacBook)

- Combine MS1, MS2, LC, and/or IM similarities
to find redundant features to collapse.

- Find long range biochemical similarity using MS2 
and combinatorial biochemical mass shifts.

- Detect isotopologues with MS1 and isotopic mass shifts.

## Installation

Use the package manager [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html) to install environment.yml for minimum requirements.

```bash
conda env create -f environment.yml
```

## Python dependencies
- python3.9
- numpy
- scipy


## Contributing
Pull requests are welcome.

For major changes, please open an issue first to discuss what you would like to change.
