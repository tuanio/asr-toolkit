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
    aux_vocab = ["<p>", "<s>", "<e>", " ", ":", "'"] + list(map(str, range(10)))

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

    def __init__(self, lang: str = "vi", **kwargs):
        super().__init__()
        self.lang = lang
        assert self.lang in ["vi", "en"], "Language not found"
        self.vocab = self.origin_vocab[lang]
        self.list_vocab = self.origin_list_vocab[lang]
        self.n_class = len(self.list_vocab)
        self.sos_id = 1
        self.eos_id = 2
        self.blank_id = 0

    def tokenize(self, s: str) -> List:
        return list(s)

    def text2int(self, s: str) -> torch.Tensor:
        return torch.Tensor([self.vocab[i] for i in s])

    def int2text(self, s: torch.Tensor) -> str:
        text = ""
        for i in s:
            if i in [self.sos_id, self.blank_id]:
                continue
            if i == self.eos_id:
                break
            text += self.list_vocab[i]
        return text


class BPEBased(TextProcess):
    def __init__(
        self,
        vocab_size=8192,
        pct_bpe=0.2,
        word_tokenizer=None,
        silent=True,
        ngram_min=2,
        ngram_max=2,
        **kwargs
    ):
        super().__init__()
        self.eow = "<e>"
        self.sow = "<s>"
        self.pad = "<p>"
        self.encoder = bpe.Encoder(
            vocab_size=vocab_size,
            pct_bpe=pct_bpe,
            word_tokenizer=word_tokenizer,
            silent=silent,
            ngram_min=ngram_min,
            ngram_max=ngram_max,
            EOW=self.eow,
            SOW=self.sow,
            PAD=self.pad,
        )
        self.n_class = vocab_size

    def fit(self, text_corpus: str = ""):
        self.encoder.fit(text_corpus)
        self.blank_id = self.encoder.word_vocab[self.pad]
        self.sos_id = self.encoder.bpe_vocab[self.sow]
        self.eos_id = self.encoder.bpe_vocab[self.eow]

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
        self.blank_id = self.encoder.word_vocab[self.pad]
