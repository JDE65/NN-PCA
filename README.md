# NN-PCA
Principal Component Analysis PCA for dimensionality reduction 

Reduction of matrix X of size m*n to a matrix Z of size m*k (with k <= n). 
The tools allow to choose the size of Z based on the choice of retained variance (anywhere from 99,99% down to...)
- compute_K_PCA : The matrix reduction tool is proposed as a separate tool that computes Z and provides the retained variance for each value of K, from 1 to n
- nn_PCA : is a single hidden neural network that applies the reduction of dimensionality of X from n to k
- nn_PCAlambda : is a tool that proposes the cost function J based on a set of different values of Lambda and a reduction of X to Z
