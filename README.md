# BLINK

Blur and Link (BLINK) is a Python library that acts 
as an abstraction of sparse matrices to enable fast 
and efficient cosine scoring across noisy dimensions. 
Originally designed for fragmentation mass spectra, 
BLINK maintains capabilities relevent to mass spec
such as neutral-losses and mass shifts.

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

## Math
The BLINK vectors are triplets of real numbers which represent a value at a specified 2d coordinate.
Vector addition is accomplished by simply appending vectors.
Vector multiplication is defined as:
    X<sub>ij</sub>Y<sub>lm</sub> = (K(j,l)XY)<sub>im</sub>
where K is a kernel function describing the similarity of j and l.

For K(x,y) = {1 if x==y else 0} where x,y are natural numbers, 
BLINK vector multiplication behaves like matrix multiplication 
where coordinates act as rows and columns.


## Contributing
Pull requests are welcome.

For major changes, please open an issue first to discuss what you would like to change.
