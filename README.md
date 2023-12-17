# Emulator-based Component Analysis (ECA)

This pytorch class implements Emulator-based component analysis (ECA) as described in J. Niskanen, A. Vladyka, J. Niemi, and C. J. Sahle, “Emulator-based decomposition for structural sensitivity of core-level spectra,” Royal Society Open Science, vol. 9, 2022, doi: [10.1098/rsos.220093](https://doi.org/10.1098/rsos.220093).

Idea:

0. For a given trained predictor Emu: X -> Y (e.g., neural network)
1. We define N basis vectors V in X space, such that the covered variance for the prediction of the X, projected on V space, is maximized in respect to Y.
2. These vectors are calculated one by one, i.e. projections on the first vector cover most of the variance.

Similar concept: see [w:Projection pursuit regression](https://en.wikipedia.org/wiki/Projection_pursuit_regression)

## Glossary and functions:

__V__ - space of ECA vectors

__fit(x, y)__ calculation of the V basis 

__transform__ x -> t, t = x @ V.T - scores of X in V space (i.e. length of the projections of X on V space)

__expand__ t -> x', x' = t @ V - projection of X data to V space 

__project__ x -> x', x' = expand(transform(x))

__inverse__ y -> t', optimization search of the t-scores for the given Y

__reconstruct__ y -> (t') -> x', via _inverse_

## Requirements

    pytorch (>2.0), tqdm
    
## Usage

    from eca import ECA

    # define options
    options = {
        'tol'   : 1E-5, 
        'epochs': 5000, 
        'lr'    : 1E-3, 
        'batch_size': 512,
        'seed' : 123
    }
    # assuming model is a torch-based (instance of nn.Module) neural network to predict X -> Y
    eca = ECA(model) 
    # run evaluation of the vectors
    # data_x, data_y are torch.Tensor(torch.float32) tensors
    v, y_var, x_var = eca.fit(data_x, data_y, n_comp=3, options=options, verbose=True)

To run inverse transformation (find an approximation in V-space for a given y):    

    # inverse transformation
    options_inv = {
        'tol'   : 1E-4, 
        'epochs': 1000, 
        'lr'    : 5E-2, 
    }
    t = eca.inverse(y, options=options_inv, verbose=True)
