from pydub import AudioSegment
import os

folder = "dataset/"

for file in os.listdir(folder):
    file_path = os.path.join(folder, file)
    name, ext = os.path.splitext(file)
    if ext == ".wav":
        continue
    else:
        output_file = f'{folder}{name}.wav'
        sound = AudioSegment.from_mp3(file_path)
        sound.export(output_file, format="wav")
        os.remove(file_path)

for file in os.listdir(folder):
    file_path = os.path.join(folder, file)
    name, ext = os.path.splitext(file)

    stereo_audio = AudioSegment.from_file(file_path, format="wav")
    mono_audios = stereo_audio.split_to_mono()
    mono_left = mono_audios[0].export(f'{folder}mono_{name}{ext}', format="wav")
    os.remove(file_path)
