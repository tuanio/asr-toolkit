# ASR E2E Toolkit


## Encoder output
- Input: `inputs`, `input_lengths`
- Output: `outputs`, `output_lengths`

## Decoder:
- Input: `targets`, `target_lengths`, `encoder_outputs` (optional)
- Output: `decoder_outputs`

- Nếu có `encoder_outputs` thì mới sử dụng attention

Với mỗi lớp `encoder` và `decoder` thì đều phải có tham số `output_dim`

## References
- https://github.com/sooftware/conformer

## Variable
set HYDRA_FULL_ERROR=1

## Run
- `python main.py -cp conf -cn configs`

## Training Template
- **Conformer-Transformer-AED Vivos (WER: 10%):** https://colab.research.google.com/drive/10dpTTy5huj7SjLst2Jj_Iff2aXLYJoBS?usp=sharing
- **Conformer-Transformer-AED LibriSpeech (WER 0%):** https://www.kaggle.com/code/tuannguyenvananh/conformer-transformer-aed-librispeech

## Install package

conda create -n train_env -c pytorch-nightly -c nvidia -c pytorch -c conda-forge python=3.7 pytorch torchaudio cudatoolkit pandas numpy ipykernel