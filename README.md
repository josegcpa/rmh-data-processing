Requirements:

- Be mindful this was compiled to use `torch` with a specific CUDA version/driver (CUDA 11.8 and NVIDIA driver version 470)
- Installing `boost` (`sudo apt-get install libboost-all-dev` or `conda install -c conda-forge boost`) as this is required by `lightgbm` when compiled to use GPU.
- 