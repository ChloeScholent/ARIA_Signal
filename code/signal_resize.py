import os
import ffmpeg

folder = "dataset/"
new_folder = "trimmed_audio/"

os.makedirs(new_folder, exist_ok=True)

for file in os.listdir(folder):
    if not file.lower().endswith(('.wav', '.mp3', '.flac')):  # Skip non-audio files
        continue

    file_path = os.path.join(folder, file)
    new_file_path = os.path.join(new_folder, file)

    # Load the audio
    audio_stream = ffmpeg.input(file_path)

    # Trim to 130 seconds
    trimmed_stream = audio_stream.filter('atrim', end=4)

    # Output the result
    (
        ffmpeg
        .output(trimmed_stream, new_file_path)
        .overwrite_output()  # Allow overwriting existing files
        .run(quiet=True)
    )

