import torch
from torch import nn, Tensor
from asr_toolkit.data.dataset import VivosDataset ,ComposeDataset
from asr_toolkit.data.datamodule import DataModule
from asr_toolkit.text import CharacterBased, BPEBased
from asr_toolkit.encoder import Conformer, VGGExtractor
from asr_toolkit.framework import CTCModel, AEDModel, RNNTModel, JointCTCAttentionModel
import pytorch_lightning as pl
# import hydra
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
        self.output_dim = layers[-1].output_dim

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

        # create dataset
        assert cfg.text.selected in cfg.text.all_types, "Dataset not found!"
        if cfg.dataset.selected == "vivos":
            train_set = VivosDataset(**cfg.dataset.hyper.vivos, subset="train")
            test_set = VivosDataset(**cfg.dataset.hyper.vivos, subset="test")
            val_set = test_set
            predict_set = test_set

        # create text process
        assert cfg.text.selected in cfg.text.all_types, "Text Process based not found!"
        if cfg.text.selected == "char":
            text_process = CharacterBased(**cfg.text.hyper.char)
        elif cfg.text.selected == "bpe":
            text_process = BPEBased(**cfg.text.hyper.bpe)
        n_class = text_process.n_class

        # create data module
        dm = DataModule(
            train_set,
            val_set,
            test_set,
            predict_set,
            text_process,
            cfg.general.batch_size,
        )

        cfg_model = cfg.model

        # create encoder and decoder
        encoder = Encoder(cfg_model.encoder)
        decoder = None
        if cfg_model.decoder.selected == "lstm":
            decoder = ...
        elif cfg_model.decoder.selected == "transformer":
            decoder = ...

        # create framework
        framework_cfg_dict = dict(
            encoder=encoder,
            decoder=decoder,
            n_class=n_class,
            cfg_model=cfg_model,
            text_process=text_process,
        )
        if not decoder:  # is None
            del framework_cfg_dict["decoder"]

        assert (
            cfg_model.framework.selected in cfg_model.framework.all_types
        ), "Framework not found!"
        if cfg_model.framework.selected == "ctc":
            framework = CTCModel(**framework_cfg_dict, **cfg_model.framework.hyper.ctc)
        elif cfg_model.framework.selected == "aed":
            framework = AEDModel(**framework_cfg_dict, **cfg_model.framework.hyper.aed)
        elif cfg_model.framework.selected == "rnnt":
            framework = RNNTModel(
                **framework_cfg_dict, **cfg_model.framework.hyper.rnnt
            )
        elif cfg_model.framework.selected == "joint_ctc_attention":
            framework = JointCTCAttentionModel(
                **framework_cfg_dict, **cfg_model.framework.hyper.joint_ctc_attention
            )

        # logger
        tb_logger = pl.loggers.tensorboard.TensorBoardLogger(**cfg.trainer.tb_logger)
        lr_monitor = pl.callbacks.LearningRateMonitor(**cfg.trainer.lr_monitor)

        trainer = pl.Trainer(logger=tb_logger, callbacks=[lr_monitor], **cfg.trainer.hyper)
        trainer.fit(model=framework, datamodule=dm)

   

main()
