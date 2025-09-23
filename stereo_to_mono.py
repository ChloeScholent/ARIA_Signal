from pydub import AudioSegment
import os

stereo_files = ["XC702143.wav", 'XC972996.wav']

for file in stereo_files:
    extension = os.path.splitext(file)[1].lstrip(".")
    filename = os.path.splitext(os.path.basename(file))[0]
    stereo_audio = AudioSegment.from_file(file, format=extension)
    mono_audios = stereo_audio.split_to_mono()
    mono_left = mono_audios[0].export(f'{filename}_left.wav', format='wav')
    mono_right = mono_audios[1].export(f'{filename}_right.wav', format='wav')



