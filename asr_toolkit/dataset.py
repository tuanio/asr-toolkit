from ctypes import util
import os
from sklearn import utils
import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
from pathlib import Path
from utils import mp3ToWav

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
    '''
    có thể sài cho FPT or dataset have structure
    |folder
    |____script.csv
    |____file1.wav
    |____***
    |____fileN.wav         
    '''
    def __init__(self,root:str="",n_fft: int = 200):
        super().__init__()
        self.root = root
       
        script = list(Path(root).glob("*.csv"))
        assert script != [], "khong tim thay script"
        transcript =  pd.read_csv(script[0])
        transcript.name = transcript.name.apply(lambda x: Path(self.root,x))
        transcript.name = transcript.name.apply(lambda x:Path(str(x)[:-3]+"wav"))
        
        self.transcript = transcript[['name','trans']]
        self.wav = list(Path(self.root).glob("*.wav"))
        if self.wav ==[]:
            print("không có .wav trong path chọn định dạng .mp3")
            mp3 = list(Path(self.root).glob("*.mp3"))
            self.transcript = transcript[transcript['name'] in mp3 ]
            self.transcript.name =  [mp3ToWav(mp3_path) for mp3_path in mp3] #conver and add new path
            print(self.transcript.name)
        self.feature_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft)
        
    def __len__(self):
        return len(self.wav)
    
    def __getitem__(self, index:int) :
        filepath, trans = self.transcript.iloc[index].values
        wave, sr = torchaudio.load(filepath)
        specs = self.feature_transform(wave)  # channel, feature, time
        specs = specs.permute(0, 2, 1)  # channel, time, feature
        specs = specs.squeeze()
        return specs, trans
class  VNPostCastDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
        use this class for dataset have structure
        folder
        |__chunks_audio
        |   |__folderPostcast1
        |   |   |__file.wav
        |   |   |__... 
        |   |__ . . . 
        |       
        |__transcripts
            |__transforFolder1.csv
            |__...
    """
    def __init__(self, root:str="",n_fft: int = 200 ):
        super().__init__()
        self.root = root
        self.walker = None
        self.csv = list(Path(root).glob("*/*.csv"))
        path = [p.parts[-1][:-4] for p in self.csv]
        print(path[0])
        self.wav = list(Path(root).glob(r"{}".format(path[0])))
        print(self.wav)
        
        self.feature_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft)

class YoutobeDataset(Dataset):
    '''
    có thể sài cho FPT or dataset have structure
    |folder
    |__folder     
    |    |__file1.wav
    |    |__***
    |    |__fileN.wav   
    |__folder
        |__script.csv      
    '''
    def __init__(self,root:str="",n_fft: int = 200):
        super().__init__()
        self.root = root
       
        script = list(Path(root).glob("*.csv"))
        assert script != [], "khong tim thay script"
        transcript =  pd.read_csv(script[0])
        transcript.name = transcript.name.apply(lambda x: Path(self.root,x))
        transcript.name = transcript.name.apply(lambda x:Path(str(x)[:-3]+"wav"))
        
        self.transcript = transcript[['name','trans']]
        self.wav = list(Path(self.root).glob("*/*.wav"))
        if self.wav ==[]:
            print("không có .wav trong path chọn định dạng .mp3")
            mp3 = list(Path(self.root).glob("*/*.mp3"))
            self.transcript = transcript[transcript['name'] in mp3 ]
            self.transcript.name =  [mp3ToWav(mp3_path) for mp3_path in mp3] #conver and add new path
            print(self.transcript.name)
        self.feature_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft)
        
    def __len__(self):
        return len(self.wav)
    
    def __getitem__(self, index:int) :
        filepath, trans = self.transcript.iloc[index].values
        wave, sr = torchaudio.load(filepath)
        specs = self.feature_transform(wave)  # channel, feature, time
        specs = specs.permute(0, 2, 1)  # channel, time, feature
        specs = specs.squeeze()
        return specs, trans
        
class ComposeDataset(Dataset):
    """
        this dataset aim to load:
            - vivos
            - vin big data
            - vietnamese podcasts
    """

    def __init__(
        self,
        vivos_root: str = "",
        vivos_subset: str = "train",
        vlsp_root: str = "",
        podcasts_root: str = "",
        n_fft: int = 400,
    ):

        super().__init__()
        self.feature_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft)

        self.walker = self.init_vivos(vivos_root, vivos_subset)

        if vivos_subset == "train":
            self.walker += self.init_vlsp(vlsp_root)

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
    def init_FPT(self,root):
        pass

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        filepath, trans = self.walker[idx]

        wave, sr = torchaudio.load(filepath)

        specs = self.feature_transform(wave)  # channel, feature, time
        specs = specs.permute(0, 2, 1)  # channel, time, feature
        specs = specs.squeeze()  # time, feature

        return specs, trans

if __name__ == '__main__': 
    path = "D:/2022/Python/ARS/data/vietnamese_postcast"
    vn = VNPostCastDataset(path)
