import torch
from torch import nn, Tensor
from asr_toolkit.data.dataset import VivosDataset
from asr_toolkit.data.datamodule import DataModule
from asr_toolkit.text import CharacterBased, BPEBased
from asr_toolkit.encoder import Conformer, VGGExtractor
import pytorch_lightning as pl
import hydra
from omegaconf import OmegaConf, DictConfig
import argparse
from typing import Tuple


class Encoder(nn.Module):
    def __init__(self, cfg_encoder: DictConfig):
        super().__init__()
        input_dim = cfg_encoder.hyper.general.input_dim
        layers = []
        for structure in cfg_encoder.structure:
            assert structure in cfg_encoder.all_types, "Encoder structure not found!"

            if structure == "conformer":
                encoder = Conformer(**cfg_encoder.hyper.conformer, input_dim=input_dim)
            elif structure == "vgg":
                encoder = VGGExtractor(**cfg_encoder.hyper.vgg, input_dim=input_dim)
            elif structure == "lstm":
                encoder = ...
            elif structure == "transformer":
                encoder = ...

            input_dim = encoder.output_dim

            layers.append(encoder)

        self.layers = nn.ModuleList(layers)

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
            input
                inputs: batch of spectrogram
                input_lengths: length of each spectrogram
            output
                outputs, output_lengths
        """
        for layer in self.layers:
            inputs, input_lengths = layer(inputs, input_lengths)
        return inputs, input_lengths


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Config path")
    parser.add_argument("-cp", help="config path")  # config path
    parser.add_argument("-cn", help="config name")  # config name

    args = parser.parse_args()

    @hydra.main(config_path=args.cp, config_name=args.cn)
    def main(cfg: DictConfig):

        assert cfg.text.selected in cfg.text.all_types, "Dataset not found!"
        if cfg.dataset.selected == "vivos":
            train_set = VivosDataset(**cfg.dataset.hyper.vivos, subset="train")
            test_set = VivosDataset(**cfg.dataset.hyper.vivos, subset="test")
            val_set = test_set
            predict_set = test_set

        assert cfg.text.selected in cfg.text.all_types, "Text Process based not found!"
        if cfg.text.selected == "char":
            text_process = CharacterBased(**cfg.text.hyper.char)
        elif cfg.text.selected == "bpe":
            text_process = BPEBased(**cfg.text.hyper.bpe)
        n_class = text_process.n_class

        dm = DataModule(
            train_set,
            test_set,
            val_set,
            predict_set,
            text_process,
            cfg.general.batch_size,
        )

        # inputs, input_lengths, targets, target_lengths = next(iter(dm.train_dataloader()))

        encoder = Encoder(cfg.model.encoder)

    main()
