# ASR E2E Toolkit


## Encoder output
- Input: `inputs`, `input_lengths`
- Output: `outputs`, `output_lengths`

## Decoder:
- Input: `targets`, `target_lengths`, `encoder_outputs` (optional)
- Output: `decoder_outputs`

- Nếu có `encoder_outputs` thì mới sử dụng attention