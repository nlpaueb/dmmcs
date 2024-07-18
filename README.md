# DMMCS: A Data-driven Guided Decoding Mechanism for Diagnostic Captioning
Distance from Median Maximum Cosine Similarity *(DMMCS)*

This repository contains the official codebase for DMMCS, our novel data-driven guided decoding algorithm featured in **ACL Findings 2024**. You can find a pre-print of our paper "A data-driven guided decoding mechanism for Diagnostic Captioning" [here](https://arxiv.org/abs/2406.14164). *DMMCS* stands for Distance from Median Maximum Cosine Similarity.

## Installation

To get started with our framework, follow these steps to clone the repository and install the required packages. We recommend using a virtual environment for package installation to ensure a clean and isolated setup.

### Step 1: Clone the repository

```
git clone https://github.com/nlpaueb/dmmcs.git
cd dmmcs
```

### Step 2: Create and activate a virtual environment

We have tested our framework for both Conda and Virtualenv environments.

#### Conda

```
conda create -n dmmcs_venv python=3.9
conda activate dmmcs_venv
pip install -r requirements.txt
```

#### Virtualenv

```
virtualenv dmmcs_venv
source dmmcs_venv/bin/activate
pip install -r requirements.txt
```

## Usage
### Step 1: Calculate your data-specific stats

First, you need to download the ```en_core_web_sm``` package from the ```spacy``` library.

```
python -m spacy download en_core_web_sm
```

Then, you have to run the stats_extraction.py script.

```
python3 utils/stats_extraction.py --config config/stats_extractor_config.json
```

Please make sure to adjust the ```config/stats_extractor_config.json``` in order to match your local file directories.


This script generates four files that will be needed for our guided-decoding mechanism. These files can be found under the ```snapshots/artifacts``` directory.

### Step 2: Run training and/or inference

You can train and/or infer from an InstructBLIP model using the proposed guided-decoding mechanism with:

```
python3 instructBLIP-ft.py --config ../config/config.json
```

Please make sure to adjust the ```config/config.json``` args file to your own local paths and directories.

Set the ```do_dmmcs``` option equal to ```True``` in order to use the dmmcs guided-decoding mechanism during inference instead of the vanilla beam search.

## Licence

This repository is licensed under the MIT license. See [LICENSE](https://github.com/nlpaueb/dmmcs/blob/main/LICENSE) for more details.

## Contact

For any questions, inquiries or suggestions, please feel free to reach out at `pkaliosis@aueb.gr` and/or `annis@aueb.gr`.


## Citation

If you would like to use our work, please cite us using the following bibtex reference:

```
@inproceedings{dmmcs,
  title={A data-driven guided decoding mechanism for Diagnostic Captioning},
  author={Panagiotis Kaliosis and John Pavlopoulos and Foivos Charalampakos and Georgios Moschovis and Ion Androutsopoulos},
  booktitle={The 62nd Annual Meeting of the Association for Computational Linguistics},
  year={2024},
}
```
