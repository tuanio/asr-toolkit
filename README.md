# ASR-Toolkit

[![Stars](https://img.shields.io/github/stars/tuanio/asr-toolkit?style=social)](https://github.com/tuanio/asr-toolkit/stargazers)
[![Fork](https://img.shields.io/github/forks/tuanio/asr-toolkit?style=social)](https://github.com/tuanio/asr-toolkit/network/members)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-orange.svg)](https://www.pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2F978--981--19--8069--5__53-blue)](https://doi.org/10.1007/978-981-19-8069-5_53)

An End-to-End Speech Recognition Toolkit with Hydra and PyTorch Lightning

## 🔍 Overview

ASR-Toolkit is a flexible and modular framework for developing end-to-end automatic speech recognition systems. The toolkit implements state-of-the-art architectures like Conformer and Transformer models, supporting both Vietnamese and English speech recognition tasks.

Key features:
- Modular components for encoders and decoders
- Support for various attention mechanisms
- Flexible configuration via Hydra
- PyTorch Lightning integration for efficient training
- Pre-configured training templates for common datasets

## 🧩 Architecture

The toolkit is built around two main components:

### Encoder
- **Input**: `inputs`, `input_lengths`
- **Output**: `outputs`, `output_lengths`
- Transforms audio features into higher-level representations

### Decoder
- **Input**: `targets`, `target_lengths`, `encoder_outputs` (optional)
- **Output**: `decoder_outputs`
- Converts encoder outputs into text
- Attention mechanism is used when `encoder_outputs` is provided

Each `encoder` and `decoder` requires an `output_dim` parameter.

## 📁 Repository Structure

```
asr_toolkit/
├── .vscode/               # VS Code configuration
├── asr_toolkit/           # Main package
│   ├── data/              # Data loading and processing
│   ├── __init__.py
│   ├── activation.py      # Activation functions
│   ├── attention.py       # Attention mechanisms
│   ├── convolution.py     # Convolutional layers
│   ├── decoder.py         # Decoder implementations
│   ├── embedding.py       # Embedding layers
│   ├── encoder.py         # Encoder implementations
│   ├── feed_forward.py    # Feed-forward networks
│   ├── framework.py       # Training framework
│   ├── loss.py            # Loss functions
│   ├── modules.py         # Reusable modules
│   ├── text.py            # Text processing utilities
│   └── utils.py           # Helper functions
├── conf/                  # Hydra configuration files
├── .gitignore
├── README.md
├── main.py                # Entry point for training
└── requirements.txt       # Dependencies
```

## 🚀 Getting Started

### Installation

```bash
# Create conda environment
conda create -n asr_env -c pytorch-nightly -c nvidia -c pytorch -c conda-forge python=3.7 pytorch torchaudio cudatoolkit pandas numpy

# Activate environment
conda activate asr_env

# Install package
pip install -e .
```

### Usage

To run training with the default configuration:

```bash
python main.py -cp conf -cn configs
```

For custom configurations:

```bash
python main.py -cp custom_conf -cn configs
```

Set environment variable for detailed error messages:

```bash
export HYDRA_FULL_ERROR=1
```

## 📚 Training Templates

Ready-to-use Colab/Kaggle notebooks for different datasets:

- [Conformer-Transformer-AED for Vivos (Vietnamese)](https://colab.research.google.com/drive/10dpTTy5huj7SjLst2Jj_Iff2aXLYJoBS?usp=sharing)
- [Conformer-Transformer-AED for LibriSpeech (English)](https://www.kaggle.com/code/tuannguyenvananh/conformer-transformer-aed-librispeech)

## 🔧 Supported Models

- **Encoders**: Conformer, Transformer
- **Decoders**: Transformer, RNN
- **End-to-End Architectures**: 
  - Attention-based Encoder-Decoder (AED)
  - Connectionist Temporal Classification (CTC)
  - RNN-Transducer (coming soon)

## 🔬 Experimental Results

Results from the paper "[A Novel Approach for Vietnamese Speech Recognition Using Conformer](https://doi.org/10.1007/978-981-19-8069-5_53)":

| Model | Dataset | WER (%) | CER (%) |
|-------|---------|---------|---------|
| Conformer + CTC | VLSP+Vivos | 20.0 | - |
| Conformer + Transformer | LibriSpeech test-clean | 5.2 | 1.7 |
| Conformer + Transformer | Vivos test | 7.1 | 2.4 |

The research demonstrated that the Conformer model trained with CTC achieved good results with Vietnamese speech recognition, establishing a foundation for future improvements.

## 📝 Citations

If you use this toolkit for your research, please cite:

```bibtex
@InProceedings{10.1007/978-981-19-8069-5_53,
author="Van Anh Tuan, Nguyen
and Hoa, Nguyen Thi Thanh
and Dat, Nguyen Thanh
and Tuan, Pham Minh
and Truong, Dao Duy
and Phuc, Dang Thi",
editor="Dang, Tran Khanh
and K{\"u}ng, Josef
and Chung, Tai M.",
title="A Novel Approach for Vietnamese Speech Recognition Using Conformer",
booktitle="Future Data and Security Engineering. Big Data, Security and Privacy, Smart City and Industry 4.0 Applications",
year="2022",
publisher="Springer Nature Singapore",
address="Singapore",
pages="723--730",
abstract="Research on speech recognition has existed for a long time, but there is very little research on applying deep learning to Vietnamese language speech recognition. In this paper, we solve the Vietnamese speech recognition problem by deep learning speech recognition frameworks including CTC and Joint CTC/Attention combined with encoder architectures Conformer. Experimental results achieved moderate accuracy using over 115 h of training data of VLSP and Vivos. Compared with the other models, the training results show that the Conformer model trained on CTC achieved good results with a WER value of 20{\%}. Training on big data gives remarkable results and is the basis for us to continue improving the model and increasing accuracy in the future.",
isbn="978-981-19-8069-5",
doi="10.1007/978-981-19-8069-5_53"
}
```

You can also cite the toolkit itself:

```bibtex
@software{asr_toolkit,
  author = {Nguyen, Tuan},
  title = {ASR-Toolkit: End-to-End Speech Recognition with Hydra and PyTorch Lightning},
  url = {https://github.com/tuanio/asr-toolkit},
  year = {2023},
}
```

## 🔗 References

- [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)
- [Implementation reference](https://github.com/sooftware/conformer)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
