from asr_toolkit import *
from asr_toolkit.dataset import ComposeDataset
from asr_toolkit.utils import *
if __name__ == "__main__":
    print("run")
    fn =  ComposeDataset(fpt_root=r'D:\2022\Python\ARS\data\FPTOpenData',podcasts_root=r'D:\2022\Python\ARS\data\vietnamese_podcast',self_record_root=r'D:\2022\Python\ARS\data\nlp_speech_record'
    )
    print(len(fn))
    # vn = VNpodcastDataset(r'D:\2022\Python\ARS\data\vietnamese_podcast')
    # print(len(vn[:1]))