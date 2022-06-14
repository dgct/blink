# BLINK

Blur and Link (BLINK) is a Python library that acts 
as an abstraction of sparse matrices to enable fast 
and efficient cosine scoring across noisy dimensions. 
Originally designed for fragmentation mass spectra, 
BLINK maintains capabilities relevent to mass spec
such as neutral-losses and mass shifts.

## Use cases:

- Quickly score experimental MS2 against spectral libraries.
(~20 million comparisons/second on 2021 MacBook)

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

That said, BLINK's dependencies likely exist in most production environments already.

## Python dependencies
- python3.9
- numpy
- scipy

## Math
BLINK vectors are comprised of triplets of real numbers which represent values at 2d coordinates.
Vector addition is defined as vector concatenation while vector multiplication is defined as:
$$ X_{ij}*Y_{lm} = K(j,l)(X*Y)_{im} $$
where K is a kernel function describing the similarity between j and l.
In other words, BLINK "blurs" values together that are "linked" in coordinate space.

Matrix multiplication is a special case of BLINK vector multiplication when $K(x,y) = \begin{cases} 1 \text{ if } x=y \\ 0 \text{ otherwise} \end{cases}$ and coordinates x and y are natural numbers.


## Contributing
Pull requests are welcome.

For major changes, please open an issue first to discuss what you would like to change.
