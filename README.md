# Room transfer function reconstruction using complex-valued neural networks


Accompanying code to the paper  _Room transfer function reconstruction using complex-valued neural networks Networks_
[[1]](#references).

- [Dependencies](#dependencies)
- [Data Generation](#data-generation)
- [Network Training](#network-training)
- [Results Computation](#results-computation)

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
>[1] Ronchini F., Comanducci L., Pezzoli M., Antonacci F. & Sarti A., Room transfer function reconstruction using complex-valued neural networks Networks, submitted to _ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_

>[2] Lluis, F., Martinez-Nuevo, P., Bo Møller, M., & Ewan Shepstone, S. (2020). Sound field reconstruction in rooms: Inpainting meets super-resolution. The Journal of the Acoustical Society of America, 148(2), 649-659.
