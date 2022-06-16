from pathlib import Path
import pydub

def mp3ToWav(path):
    try:
        sound = pydub.AudioSegment.from_mp3(path)
        path = str(path)
        new_path =path[:-3] + 'wav'
        sound.export(new_path, format="wav")
        return Path(new_path)
    except:
        print(path + "khong the conver to wav")
