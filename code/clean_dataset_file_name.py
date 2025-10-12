#Cleaning the recorded data by keeping only the id code

import os

folder = "dataset/"

for filename in os.listdir(folder):
    old_path = os.path.join(folder, filename)
    name, ext = os.path.splitext(filename)

    if " " in name:
        new_name = name.split(" ")[0] + ext
    else:
        new_name = filename

    new_path = os.path.join(folder, new_name)

    if not os.path.exists(new_path):
        os.rename(old_path, new_path)
        print(f"Renommé : {filename} -> {new_name}")
    else:
        print(f"⚠️ Fichier déjà existant : {new_name}, renommage ignoré.")
