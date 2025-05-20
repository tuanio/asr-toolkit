# End-to-End ASR-Toolkit

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

### Model Information

| Model | Parameters | Minutes/Epoch | No. Epochs |
|-------|------------|---------------|------------|
| Conformer - CTC | 4.7M | 36 | 90 |
| Conformer - Transformer - Joint CTC/Attention | 4.9M | 41 | 55 |

### Comparison with Other Vietnamese ASR Systems

| Model | Test Dataset | WER (%) |
|-------|-------------|---------|
| Conformer - CTC (ours) | Vivos | 20.0 |
| Conformer - Transformer - Joint CTC/Attention (ours) | Vivos | 44.0 |
| --- | --- | --- |
| Wave2vec 2.0 [2] | Vivos | 6.15 |
| Google [3] | News | 22.41 |
| Data Augmentation [4] | Realistic voice | 10.3 |
| TDNN + LSTM (E2E Model) [1] | VLSP2018 + FPT | 9.71 |
| Conformer + Transformer | LibriSpeech test-clean | 5.2 |
| Conformer + Transformer | Vivos test | 7.1 |

### Sample Predictions (Vietnamese)

Here are sample predictions from the models compared to ground truth:

**Sample 1:** "and when I saw that I was still steadfast in replying"

| Model | Ground Truth | Prediction |
|-------|-------------|------------|
| Conformer - CTC | và khi thấy tôi vẫn kiên định trả lời trớt quớt | và khi thấy tôi vẫn kiên định trả lời trớt quốc |
| Conformer - Transformer - Joint CTC/Attention | và khi thấy tôi vẫn kiên định trả lời trớt quớt | và khi thấy tôi vẫn kiên định trả lời trớt quốc |

**Sample 2:** "the brother was then allowed to bid to build the canal but did not proceed"

| Model | Ground Truth | Prediction |
|-------|-------------|------------|
| Conformer - CTC | người anh sau đó được phép thầu xây kênh này nhưng không tiến hành | người anh sau đó được phép thầu say kên này nhưng không tê hàn |
| Conformer - Transformer - Joint CTC/Attention | người anh sau đó được phép thầu xây kênh này nhưng không tiến hành | người anh sau đó được phép phiếp thầu phay kên này nhưng không tiếng hành |

**Sample 3:** "I am afraid of losing him, but I feel insulted"

| Model | Ground Truth | Prediction |
|-------|-------------|------------|
| Conformer - CTC | em sợ mất anh ấy nhưng lại cảm thấy mình bị xúc phạm | em sớm mận anh ấy nhưng là cảm thá mình bị xúc phạm |
| Conformer - Transformer - Joint CTC/Attention | em sợ mất anh ấy nhưng lại cảm thấy mình bị xúc phạm | em sớ mất anh ấy nhưng là cảm thái mình bị xúc phải |

**Sample 4:** "can heal without leaving a scar"

| Model | Ground Truth | Prediction |
|-------|-------------|------------|
| Conformer - CTC | có thể lành lặn mà không để lại vết sẹo | có thể lảng lạng mà không bể lại dết sẹo |
| Conformer - Transformer - Joint CTC/Attention | có thể lành lặn mà không để lại vết sẹo | có thể lảng lặng mà không bệt sẹo |

The research demonstrated that the Conformer model trained with CTC achieved reasonable results with Vietnamese speech recognition (20% WER), establishing a foundation for future improvements.

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

## 📚 References

1. Nguyen, V. H.: An End-to-End Model for Vietnamese Speech Recognition. In: 2019 IEEE-RIVF International Conference on Computing and Communication Technologies (RIVF), 2019, pp. 1-6, (2019). https://doi.org/10.1109/RIVF.2019.8713758

2. Nguyen, B.: Vietnamese end-to-end speech recognition using wav2vec 2.0. (2021). https://doi.org/10.5281/zenodo.5356039

3. Thanh, N., T., M., Dung, P., X., Hay, N., N., Bich, L., N., Quy, Đ., X.: Đánh giá các hệ thống nhận dạng giọng nói tiếng Việt (Vais, Viettel, Zalo, Fpt và Google) trong bản tin. In: Tạp Chí Khoa Học Giáo Dục Kỹ Thuật, Vietnamese. vol. 63, (2021)

4. Bao, N., Q., Tuan, M., V., Trung, L., Q., Quyen, D., B., Hai, D., V.: Development of a Vietnamese Large Vocabulary Continuous Speech Recognition System under Noisy Conditions. In Proceedings of the Ninth International Symposium on Information and Communication Technology (SoICT 2018). Association for Computing Machinery, New York, NY, USA, 222–226, (2018). https://doi.org/10.1145/3287921.3287938

5. [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)

6. [Implementation reference](https://github.com/sooftware/conformer)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
