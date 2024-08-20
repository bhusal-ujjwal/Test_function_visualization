# Description
By using the C++ programming language and CUDA technology, implement the application according to the following specifications:

## Build the first kernel.
- The kernel will take an already allocated matrix of numbers with a sufficient number of elements as input parameters.
- The kernel will also take other parameters, such as min, max, step, and so on, as input parameters.
- The function implemented within the kernel will generate matrix values based on a predefined test optimization function.

## Build the second kernel.
- The kernel converts the matrix from the previous input point, and based on this matrix, it generates a PNG image.
- The kernel also takes the minimum and maximum values found in the matrix as parameters.
- The transformation of values into a color scale (normalization of values) is arbitrary (you can use black/white, one color, or multicolor space).

## Non-specified details above can be chosen or replaced as desired, as well as a number of auxiliary functions, kernels, and the like. Attention should be given to the correct allocation of memory and the correct synchronization of the kernels when starting them.
