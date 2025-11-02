import os
import shutil

in_folder = "annotated_dataset/"
out_folder = "dataset/"
os.makedirs(out_folder, exist_ok=True)

for filename in os.listdir(in_folder):
    old_path = os.path.join(in_folder, filename)
    name, ext = os.path.splitext(filename)

    if os.path.isdir(old_path):
        continue

    if " " in name:
        new_name = name.split(" ")[0] + ext
    else:
        new_name = filename

    new_path = os.path.join(out_folder, new_name)

    shutil.copy2(old_path, new_path)
    print(f"✅ Copied and renamed: {filename} -> {new_name}")
