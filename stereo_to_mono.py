from pydub import AudioSegment

stereo_audio = AudioSegment.from_file("Chardonneret_stereo.wav", format="wav")

mono_audios = stereo_audio.split_to_mono()

mono_left = mono_audios[0].export(
    "Chardonneret_left.wav",
    format="wav")
mono_right = mono_audios[1].export(
    "Chardonneret_right.wav",
    format="wav")