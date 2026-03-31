import os

# cấu trúc project
folders = [
    "data",
    "data/userA",
    "data/userB"
]

files = [
    "collect_data.py",
    "train_model.py",
    "recognite.py",
    "face_recognizer_model.yml",
    "label_dict.npy"
]

print("Creating folders...")

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print("Created:", folder)

print("\nCreating files...")

for file in files:
    if not os.path.exists(file):
        open(file, "w").close()
        print("Created:", file)
    else:
        print("Already exists:", file)

print("\nProject structure ready.")