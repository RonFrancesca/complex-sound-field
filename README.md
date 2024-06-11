<div align="center">

# Room Transfer Function Reconstruction Using Complex-valued Neural Networks and Irregularly Distributed Microphones

<!-- <img width="700px" src="docs/new-generic-style-transfer-headline.svg"> -->
 
[Francesca Ronchini](https://www.linkedin.com/in/francesca-ronchini/)<sup>1</sup>, [Luca Comanducci](https://lucacoma.github.io/)<sup>1</sup>, [Mirco Pezzoli](https://www.linkedin.com/in/mirco-pezzoli/)<sup>1</sup>, [Fabio Antonacci](https://www.deib.polimi.it/ita/personale/dettagli/573870)<sup>1</sup>, and [Augusto Sarti](https://www.deib.polimi.it/eng/people/details/61414)<sup>1</sup>

<sup>1</sup> Dipartimento di Elettronica, Informazione e Bioingegneria - Politecnico di Milano<br>

Accepted at European Signal Processing Conference (EUSIPCO) 2024

[![arXiv](https://img.shields.io/badge/arXiv-2402.04866-b31b1b.svg)](https://arxiv.org/abs/2402.04866)

</div>

- [Abstract](#abstract)
- [Dependencies](#dependencies)
- [Data Generation](#data-generation)
- [Network Training](#network-training)
- [Results Computation](#results-computation)

### Abstract

Reconstructing the room transfer functions needed to calculate the complex sound field in a room has several important real-world applications. However, an unpractical number of microphones is often required. Recently, in addition to classical
signal processing methods, deep learning techniques have been applied to reconstruct the room transfer function starting from
a very limited set of measurements at scattered points in the room. In this paper, we employ complex-valued neural networks
to estimate room transfer functions in the frequency range of the first room resonances, using a few irregularly distributed
microphones. To the best of our knowledge, this is the first time that complex-valued neural networks are used to estimate room transfer functions. To analyze the benefits of applying complexvalued optimization to the considered task, we compare the proposed technique with a state-of-the-art kernel-based signal processing approach for sound field reconstruction, showing that the proposed technique exhibits relevant advantages in terms of phase accuracy and overall quality of the reconstructed sound field. For informative purposes, we also compare the model with a similarly-structured data-driven approach that, however, applies a real-valued neural network to reconstruct only the magnitude of the sound field. 

### Dependencies
- Python, it has been tested with version 3.9.17
- Numpy, scikit-image, scikit-learn,argparse, tqdm, matplotlib
- Pytorch 2.0.1+cu118
- [complexPyTorch](https://github.com/wavefrontshaping/complexPyTorch)

### Data generation

Data generation code is contained in the folder _create\_dataset_ and is partially taken from [Lluis et al.](https://github.com/francesclluis/sound-field-neural-network)[[2]](#references)

### Network training
The training of the network uses the parameters contained in _config/config.jon_ and can be run using _main.py_, which takes the following arguments:
- --config: String, path to the configuration file
- --best_model_path: String, name of the best-performing model file

### Results computation
By running _test.py_ it is possible to compute the results contained in the paper. Specifically, the script computes the Normalized Mean Squared Error (NMSE) using the parameters and network model indicated by the selected configuration. The script takes the following arguments:
- --config: String, path to the configuration file
- --best_model_path: String, name of the best-performing model file

# References
>[1] Ronchini F., Comanducci L., Pezzoli M., Antonacci F. & Sarti A., Room Transfer Function Reconstruction Using Complex-valued Neural Networks and Irregularly Distributed Microphones, submitted to EUSIPCO, European Signal Processing Conference 2024. 

>[2] Lluis, F., Martinez-Nuevo, P., Bo Møller, M., & Ewan Shepstone, S. (2020). Sound field reconstruction in rooms: Inpainting meets super-resolution. The Journal of the Acoustical Society of America, 148(2), 649-659.
