import torch
import bpe
from typing import List
from abc import ABC, abstractmethod


class TextProcess(ABC):
    @abstractmethod
    def text2int(self, data):
        pass

    @abstractmethod
    def int2text(self, data):
        pass

    def __init__(self):
        ...

    def decode(self, argmax: torch.Tensor):
        """
        decode greedy with collapsed repeat
        """
        decode = []
        for i, index in enumerate(argmax):
            if index != self.blank_id:
                if i != 0 and index == argmax[i - 1]:
                    continue
                decode.append(index.item())
        return self.int2text(decode)


class CharacterBased(TextProcess):
    aux_vocab = ["<p>", "<s>", "<e>", " ", ":", "'"]

    origin_list_vocab = {
        "en": aux_vocab + list("abcdefghijklmnopqrstuvwxyz"),
        "vi": aux_vocab
        + list(
            "abcdefghijklmnopqrstuvwxyzàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
        ),
    }

    origin_vocab = {
        lang: dict(zip(vocab, range(len(vocab))))
        for lang, vocab in origin_list_vocab.items()
    }

    def __init__(self, lang: str = "vi"):
        super().__init__()
        self.lang = lang
        assert self.lang in ["vi", "en"], "Language not found"
        self.vocab = self.origin_vocab[lang]
        self.list_vocab = self.origin_list_vocab[lang]
        self.n_class = len(self.list_vocab)
        self.sos_id = 1
        self.eos_id = 2
        self.blank_id = 0

    def text2int(self, s: str) -> torch.Tensor:
        return torch.Tensor([self.vocab[i] for i in s.lower()])

    def int2text(self, s: torch.Tensor) -> str:
        return "".join([self.list_vocab[i] for i in s if i > 2])


class BPEBased(TextProcess):
    def __init__(
        self,
        vocab_size=8192,
        pct_bpe=0.2,
        word_tokenizer=None,
        silent=True,
        ngram_min=2,
        ngram_max=2,
        required_tokens=None,
        strict=False,
        lowercase=True,
    ):
        super().__init__()
        self.encoder = bpe.Encoder(
            vocab_size,
            pct_bpe,
            word_tokenizer,
            silent,
            ngram_min,
            ngram_max,
            required_tokens,
            strict,
            lowercase,
        )
        self.n_class = vocab_size

    def fit(self, text_corpus: str = ""):
        self.encoder.fit(text_corpus)
        self.blank_id = self.encoder.word_vocab["__blank"]
        self.sos_id = self.encoder.word_vocab["__sow"]
        self.eos_id = self.encoder.word_vocab["__eow"]

    def tokenize(self, text: str):
        return self.encoder.tokenize(text)

    def text2int(self, text: str):
        if isinstance(text, str):
            text = [text]
        return torch.LongTensor(next(self.encoder.transform(text)))

    def int2text(self, idx: List[int]):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        idx = [[i for i in idx if i not in [0, 1, 2]]]  # 0, 1, 2 is not use
        return next(self.encoder.inverse_transform(idx))

    def load(self, in_path):
        self.encoder = self.encoder.load(in_path)
        self.blank_id = self.encoder.word_vocab["__blank"]
