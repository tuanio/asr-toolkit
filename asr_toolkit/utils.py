from pathlib import Path
import pydub
import torchaudio


def mp3ToWav(path):
    try:
        sound = pydub.AudioSegment.from_mp3(path)
        path = str(path)
        new_path = path[:-3] + "wav"
        sound.export(new_path, format="wav")
        return Path(new_path)
    except:
        print(path + "khong the conver to wav")


def load_and_transform(audio_path: str, n_fft: int = 159):
    wave, sr = torchaudio.load(audio_path)
    feature_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft)
    specs = feature_transform(wave)
    specs = specs.permute(0, 2, 1)
    return specs # channel, time, feature