import torch
from torch import nn, Tensor
from asr_toolkit.data.dataset import (
    VivosDataset,
    ComposeDataset,
    LibriSpeechDataset,
    TimitDataset,
)
from asr_toolkit.data.datamodule import DataModule
from asr_toolkit.text import CharacterBased, BPEBased, PhonemeBased
from asr_toolkit.encoder import Conformer, VGGExtractor, LSTMEncoder, TransformerEncoder
from asr_toolkit.decoder import LSTMDecoder, TransformerDecoder
from asr_toolkit.framework import CTCModel, AEDModel, RNNTModel, JointCTCAttentionModel
from asr_toolkit.utils import load_and_transform
import pytorch_lightning as pl

import hydra
from omegaconf import OmegaConf, DictConfig
import argparse
import pickle


from typing import Tuple


class Encoder(nn.Module):
    def __init__(
        self, cfg_encoder: DictConfig, blank_id: int = None, device: str = None
    ):
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
                encoder = LSTMEncoder(**cfg_encoder.hyper.lstm, input_dim=input_dim)
            elif structure == "transformer":
                encoder = TransformerEncoder(
                    **cfg_encoder.hyper.transformer,
                    input_dim=input_dim,
                    blank_id=blank_id,
                    device=device,
                )

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


parser = argparse.ArgumentParser(description="Config path")
parser.add_argument("-cp", help="config path")  # config path
parser.add_argument("-cn", help="config name")  # config name

args = parser.parse_args()


@hydra.main(version_base="1.2", config_path=args.cp, config_name=args.cn)
def main(cfg: DictConfig):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create dataset
    assert cfg.text.selected in cfg.text.all_types, "Dataset not found!"
    if cfg.dataset.selected == "vivos":
        train_set = VivosDataset(**cfg.dataset.hyper.vivos, subset="train")
        test_set = VivosDataset(**cfg.dataset.hyper.vivos, subset="test")
        val_set = test_set
        predict_set = test_set
    elif cfg.dataset.selected == "compose":
        train_set = ComposeDataset(**cfg.dataset.hyper.compose, vivos_subset="train")
        test_set = ComposeDataset(**cfg.dataset.hyper.compose, vivos_subset="test")
        val_set = test_set
        predict_set = test_set
    elif cfg.dataset.selected == "timit":
        train_set = TimitDataset(**cfg.dataset.hyper.timit, is_test=False)
        test_set = TimitDataset(**cfg.dataset.hyper.timit, is_test=True)
        val_set = test_set
        predict_set = test_set
    elif cfg.dataset.selected == "librispeech":
        train_set = LibriSpeechDataset(
            **cfg.dataset.hyper.librispeech, subset="test-other"
        )
        val_set = LibriSpeechDataset(
            **cfg.dataset.hyper.librispeech, subset="test-other"
        )
        test_set = LibriSpeechDataset(
            **cfg.dataset.hyper.librispeech, subset="test-other"
        )
        predict_set = test_set

    print("Done setup dataset!")

    # create text process
    assert cfg.text.selected in cfg.text.all_types, "Text Process based not found!"
    if cfg.text.selected == "char":
        text_process = CharacterBased(**cfg.text.hyper.char)
    elif cfg.text.selected == "bpe":
        text_process = BPEBased(**cfg.text.hyper.bpe)

        try:
            print("Openning text corpus")
            with open("text_corpus.pkl", "rb") as f:
                text_corpus = pickle.load(f)
        except:
            print("Getting text corpus from train...")
            text_corpus = [i[1] for i in train_set]
            with open("text_corpus.pkl", "wb") as f:
                pickle.dump(text_corpus, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("Fitting text corpus to BPE...")
        text_process.fit(text_corpus)
    elif cfg.text.selected == "phoneme":
        text_process = PhonemeBased(**cfg.text.hyper.phoneme)

    n_class = text_process.n_class
    blank_id = text_process.blank_id

    cfg.model.loss.ctc.blank = blank_id
    cfg.model.loss.cross_entropy.ignore_index = blank_id
    cfg.model.loss.rnnt.blank = blank_id
    print("Done setup text!")

    # create data module
    dm = DataModule(
        train_set,
        val_set,
        test_set,
        predict_set,
        text_process,
        cfg.general.batch_size,
        **cfg.datamodule,
    )

    steps_per_epoch = len(dm.train_dataloader())
    cfg.model.lr_scheduler.one_cycle_lr.steps_per_epoch = steps_per_epoch
    print("Done setup datamodule!")

    cfg_model = cfg.model

    # create encoder and decoder
    encoder = Encoder(cfg_model.encoder, blank_id, device)
    assert (
        cfg_model.decoder.selected in cfg_model.decoder.all_types
    ), "Decoder not found!"
    if cfg_model.decoder.selected == "lstm":
        decoder = LSTMDecoder(
            **cfg_model.decoder.hyper.lstm,
            n_class=n_class,
            encoder_output_dim=encoder.output_dim,
            sos_id=text_process.sos_id,
            eos_id=text_process.eos_id,
        )
    elif cfg_model.decoder.selected == "transformer":
        decoder = TransformerDecoder(
            **cfg_model.decoder.hyper.transformer,
            n_class=n_class,
            blank_id=blank_id,
            device=device,
        )
    else:
        decoder = None
    print("Done setup encoder and decoder!")

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
        framework = RNNTModel(**framework_cfg_dict, **cfg_model.framework.hyper.rnnt)
    elif cfg_model.framework.selected == "joint_ctc_attention":
        framework = JointCTCAttentionModel(
            **framework_cfg_dict, **cfg_model.framework.hyper.joint_ctc_attention
        )
    print("Done setup framework!")

    # logger
    tb_logger = pl.loggers.tensorboard.TensorBoardLogger(**cfg.trainer.tb_logger)
    print("Done setup tb logger!")

    trainer = pl.Trainer(logger=tb_logger, **cfg.trainer.hyper)
    print("Done setup trainer!")

    ckpt_path = None
    if cfg.ckpt.use_ckpt and cfg.ckpt.ckpt_path.endswith(".ckpt"):
        ckpt_path = cfg.ckpt.ckpt_path
        framework = framework.load_from_checkpoint(ckpt_path)

    if cfg.session.train:
        trainer.fit(model=framework, datamodule=dm, ckpt_path=ckpt_path)

    if cfg.session.validate:
        trainer.validate(model=framework, datamodule=dm, ckpt_path=ckpt_path)

    if cfg.session.test:
        trainer.test(model=framework, datamodule=dm, ckpt_path=ckpt_path)

    if cfg.session.predict.is_pred:
        # print("Loading model")
        # framework = framework.load_from_checkpoint(ckpt_path)
        inputs = load_and_transform(cfg.session.predict.audio_path)
        input_lengths = torch.LongTensor([[inputs.size(1)]])
        predicts = framework.recognize(inputs, input_lengths)
        print("Predict:", *predicts)


if __name__ == "__main__":
    main()
