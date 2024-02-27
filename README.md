<!-- ABOUT THE PROJECT -->
## CR-SAVAE

Variational Autoencoders based model for Survival Analysis with Competing Risks. 

This repository provides:
* Necessary scripts to train CR-SAVAE and Deephit (using [Pycox](https://github.com/havakv/pycox) package).
* Datasets used and preprocessing functions.
* Validation metrics (C-index and IBS) adapted from PyCox.
* Code and results dictionary to replicate tables as presented in the paper.

For more details, see full paper [here]().


<!-- GETTING STARTED -->
## Getting Started
Follow these simple steps to make this project work on your local machine.

### Prerequisites
You should have the following installed on your machine:
* Python 3.8.0
* Packages in requirements.txt
  ```sh
  pip install -r requirements.txt
  ```

### Installation

Download the repo manually (as a .zip file) or clone it using Git.
   ```sh
   git clone https://github.com/Patricia-A-Apellaniz/cr-savae.git
   ```


<!-- USAGE EXAMPLES -->
## Usage

You can specify different configurations or training parameters in utils.py for CR-SAVAE and DeepHit. 

To train/test CR-SAVAE and DeepHit and show results, run the following command:
   ```sh
   python survival_analysis/main_competing.py
   ```


<!-- CONTACT -->
## Contact

Patricia A. Apellaniz - patricia.alonsod@upm.es

<p align="right">(<a href="#readme-top">back to top</a>)</p>


