from ctypes import util
from mimetypes import init
import os
from numpy import record

import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
from pathlib import Path


class LibriSpeechDataset(torch.utils.data.Dataset):

    ext_txt = ".trans.txt"
    ext_audio = ".flac"

    def __init__(
        self,
        clean_path="../input/librispeech-clean/LibriSpeech/",
        other_path="../input/librispeech-500-hours/LibriSpeech/",
        subset="train100",
        n_fft=159,
    ):
        """
        subset \in ['train100', 'train360', 'train460', 'train960', 'dev', 'test']
        """
        assert subset in [
            "train100",
            "train360",
            "train460",
            "train960",
            "dev-clean",
            "test-clean",
            "dev-other",
            "test-other",
        ], "Librispeech subset not found!"

        self.spect_func = torchaudio.transforms.Spectrogram(n_fft=n_fft)

        self.list_url = [clean_path + "train-clean-100"]

        if subset == "train360":
            self.list_url = [clean_path + "train-clean-360"]
        elif subset == "train460":
            self.list_url += [clean_path + "train-clean-360"]
        elif subset == "train960":
            self.list_url += [clean_path + "train-clean-360"]
            self.list_url += [other_path + "train-clean-360"]
        else:
            self.list_url = [clean_path + subset]

        sep = os.sep
        self._walker = []
        for path in self.list_url:
            files_path = f"*{sep}*{sep}*" + self.ext_audio
            walker = [(str(p.stem), path) for p in Path(path).glob(files_path)]
            self._walker.extend(walker)

    def __len__(self):
        return len(self._walker)

    def __getitem__(self, n):
        fileid, path = self._walker[n]
        return self.load_librispeech_item(fileid, path)

    def load_librispeech_item(self, fileid, path):
        """
        transform audio pack to spectrogram
        """

        speaker_id, chapter_id, utterance_id = fileid.split("-")

        file_text = speaker_id + "-" + chapter_id + self.ext_txt
        file_text = os.path.join(path, speaker_id, chapter_id, file_text)

        fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
        file_audio = fileid_audio + self.ext_audio
        file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)

        # Load audio
        waveform, sample_rate = torchaudio.load(file_audio)

        # Load text
        with open(file_text) as ft:
            for line in ft:
                fileid_text, transcript = line.strip().split(" ", 1)
                if fileid_audio == fileid_text:
                    break
            else:
                # Translation not found
                raise FileNotFoundError("Translation not found for " + fileid_audio)

        spectrogram = self.spect_func(waveform)
        spectrogram = spectrogram.squeeze().permute(1, 0)

        return spectrogram, transcript


class VivosDataset(Dataset):
    def __init__(self, root: str = "", subset: str = "train", n_fft: int = 200):
        super().__init__()
        self.root = root
        self.subset = subset
        assert self.subset in ["train", "test"], "subset not found"

        path = os.path.join(self.root, self.subset)
        waves_path = os.path.join(path, "waves")
        transcript_path = os.path.join(path, "prompts.txt")

        # walker oof
        self.walker = list(Path(waves_path).glob("*/*"))

        with open(transcript_path, "r", encoding="utf-8") as f:
            transcripts = f.read().strip().split("\n")
            transcripts = [line.split(" ", 1) for line in transcripts]
            filenames = [i[0] for i in transcripts]
            trans = [i[1] for i in transcripts]
            self.transcripts = dict(zip(filenames, trans))

        self.feature_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft)

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        filepath = str(self.walker[idx])
        filename = filepath.rsplit(os.sep, 1)[-1].split(".")[0]

        wave, sr = torchaudio.load(filepath)
        specs = self.feature_transform(wave)  # channel, feature, time
        specs = specs.permute(0, 2, 1)  # channel, time, feature
        specs = specs.squeeze()  # time, feature

        trans = self.transcripts[filename].lower()

        return specs, trans


class FPTOpenData(Dataset):

    """
    có thể sài cho FPT or dataset have structure
    |folder
    |____script.csv
    |____file1.wav
    |____***
    |____fileN.wav
    """

    def __init__(self, root: str = "", n_fft: int = 200):
        super().__init__()
        self.root = root
        script = list(Path(root).glob("*.csv"))
        assert script != [], "can't find the csv file script"
        transcript = pd.read_csv(script[0])
        transcript.name = transcript.name.apply(lambda x: Path(self.root, x))
        transcript.name = transcript.name.apply(lambda x: Path(str(x)[:-3] + "wav"))

        self.transcript = transcript[["name", "trans"]]
        self.wav = list(Path(self.root).glob("*.wav"))
        if self.wav == []:
            print(
                "can't find file .wav in path we will convert mp3 in this file to wav"
            )
            mp3 = list(Path(self.root).glob("*.mp3"))
            self.transcript = transcript[transcript["name"] in mp3]
            self.transcript.name = [
                mp3ToWav(mp3_path) for mp3_path in mp3
            ]  # conver and add new path
        self.walker = self.transcript.iloc[:].values
        self.feature_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft)

    def __len__(self):
        return len(self.wav)

    def __getitem__(self, index: int):
        filepath, trans = self.walker[index]
        wave, sr = torchaudio.load(filepath)
        specs = self.feature_transform(wave)  # channel, feature, time
        specs = specs.permute(0, 2, 1)  # channel, time, feature
        specs = specs.squeeze()
        return specs, trans


class VNpodcastDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
        use this class for dataset have structure
        folder
        |__chunks_audio
        |   |__folderpodcast1
        |   |   |__file.wav
        |   |   |__...
        |   |__ . . .
        |
        |__transcripts
            |__transforFolder1.csv
            |__...
    """

    def __init__(self, root: str = "", n_fft: int = 200):
        super().__init__()

        def make_walker(wav_folder, csv_root):
            csv = pd.read_csv(csv_root, encoding="utf-8")
            csv = csv[["chunk", "script"]]
            return [
                (os.path.join(wav_folder, chunk_), trans)
                for chunk_, trans in csv.values
            ]

        self.root = root
        self.walker = []
        self.make_walker = make_walker
        self.csv = list(Path(root).glob(r"*/*.csv"))
        path = [(p.parts[-1][:-4], p) for p in self.csv]
        self.wav = [
            (list(Path(root).glob(r"*/{}_chunks".format(p))), root_csv)
            for p, root_csv in path
        ]
        for p, r in self.wav:
            try:
                a = self.make_walker(p[0], r)
                self.walker += a
            except:
                pass
        self.feature_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft)

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, index: int):
        try:
            filepath, trans = self.walker[index]
            wave, sr = torchaudio.load(filepath)
            specs = self.feature_transform(wave)  # channel, feature, time
            specs = specs.permute(0, 2, 1)  # channel, time, feature
            specs = specs.squeeze()
            return specs, trans
        except:
            print("didn't find filepath in index: " + str(index))


class YoutubeDataset(Dataset):
    """
    use for FPT dataset or dataset have structure
    |folder
    |__folder
    |    |__file1.wav
    |    |__***
    |    |__fileN.wav
    |__folder
         |__script.csv
    """

    def __init__(self, root: str = "", n_fft: int = 200):
        super().__init__()
        self.root = root
        self.walker = []
        script = list(Path(root).glob("*/*.csv"))
        assert script != [], "can't find the csv file script"
        transcript = pd.read_csv(script[0], encoding="utf-8")
        self.transcript = transcript[["name", "trans"]]
        self.wav = list(Path(self.root).glob("*/*.wav"))
        if self.wav == []:
            print(
                "can't find file .wav in path we will convert mp3 in this file to wav"
            )
            mp3 = list(Path(self.root).glob("*/*.mp3"))
            self.transcript = transcript[transcript["name"] in mp3]
            self.transcript.name = [
                mp3ToWav(mp3_path) for mp3_path in mp3
            ]  # conver and add new path .wav
        for idx, filepath in enumerate(self.wav):
            self.walker.append(
                (
                    filepath,
                    self.transcript[self.transcript.name == filepath.parts[-1]][
                        "trans"
                    ].values[0],
                )
            )
        self.feature_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft)

    def __len__(self):
        return len(self.wav)

    def __getitem__(self, index: int):
        try:
            filepath, trans = self.walker[index]
            wave, sr = torchaudio.load(filepath)
            specs = self.feature_transform(wave)  # channel, feature, time
            specs = specs.permute(0, 2, 1)  # channel, time, feature
            specs = specs.squeeze()
            return specs, trans
        except:
            print("didn't find filepath in index: " + str(index))


class ComposeDataset(Dataset):
    """
    this dataset aim to load:
        - self record
        - FPTOpenDataset
        - vivos
        - vin big data
        - vietnamese podcasts
    """

    def __init__(
        self,
        vivos_root: str,
        vivos_subset: str = "train",
        vlsp_root: str = "",
        podcasts_root: str = "",
        fpt_root: str = "",
        self_record_root: str = "",
        youtube_root: str = "",
        n_fft: int = 159,
    ):
        super().__init__()
        self.feature_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft)
        self.walker = []
        if vivos_subset == "train":
            if vivos_root != "":
                self.walker.extend(self.init_vivos(vivos_root, vivos_subset))
            if vlsp_root != "":
                self.walker.extend(self.init_vlsp(vlsp_root))
            if podcasts_root != "":
                self.walker.extend(self.init_VNPodcast(podcasts_root))
            if fpt_root != "":
                self.walker.extend(self.init_FPT(fpt_root))
            if self_record_root != "":
                self.walker.extend(self.init_nlp_record(self_record_root))
            if youtube_root != "":
                self.walker.extend(self.init_youtube(youtube_root))
        else:
            self.walker.extend(self.init_vivos(vivos_root, vivos_subset))

    def init_vivos(self, root, subset):
        assert subset in ["train", "test"], "subset not found"

        path = os.path.join(root, subset)
        waves_path = os.path.join(path, "waves")
        transcript_path = os.path.join(path, "prompts.txt")

        # walker oof
        walker = list(Path(waves_path).glob("*/*"))

        with open(transcript_path, "r", encoding="utf-8") as f:
            transcripts = f.read().strip().split("\n")
            transcripts = [line.split(" ", 1) for line in transcripts]
            filenames = [i[0] for i in transcripts]
            trans = [i[1] for i in transcripts]
            transcripts = dict(zip(filenames, trans))

        def load_el_from_path(filepath):
            filename = filepath.name.split(".")[0]
            trans = transcripts[filename].lower()
            return (filepath, trans)

        walker = [load_el_from_path(filepath) for filepath in walker]
        return walker

    def init_vlsp(self, root):
        walker = list(Path(root).glob("*.wav"))

        def load_el_from_path(filepath):
            filename = filepath.name.split(".")[0] + ".txt"
            with open(Path(root) / filename, "r", encoding="utf-8") as f:
                trans = f.read().strip().lower()
                trans = trans.replace("<unk>", "").strip()
            return filepath, trans

        walker = [load_el_from_path(filepath) for filepath in walker]

        return walker

    def init_FPT(self, root):
        init = FPTOpenData(root)
        return init.walker

    def init_VNPodcast(self, root):
        init = VNpodcastDataset(root)
        return init.walker

    def init_nlp_record(self, root):
        init = FPTOpenData(root)
        return init.walker

    def init_youtube(self, root):
        init = YoutubeDataset(root)
        return init.walker

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        filepath, trans = self.walker[idx]
        wave, sr = torchaudio.load(filepath)
        specs = self.feature_transform(wave)  # channel, feature, time
        specs = specs.permute(0, 2, 1)  # channel, time, feature
        specs = specs.squeeze()  # time, feature
        return specs, trans


class TimitDataset(Dataset):
    def __init__(
        self, data_root: str, csv_path: str, n_fft: int = 159, is_test: bool = False
    ):
        super().__init__()
        df = pd.read_csv(csv_path, index_col=0)
        df = df[df.is_test == is_test].drop("is_test", axis=1)
        df.path = df.path.apply(lambda x: data_root + os.sep + x)
        df.trans = df.trans.str.split("|")

        self.walker = df.to_dict("records")
        self.feature_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft)

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        item = self.walker[idx]
        trans = item["trans"]
        wave, sr = torchaudio.load(item["path"])
        specs = self.feature_transform(wave)
        specs = specs.permute(0, 2, 1)
        specs = specs.squeeze()
        return specs, trans