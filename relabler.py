import os
import glob

def relabel_dataset_images(emotion_name, dataset_dir="dataset"):
    emotion_folder = os.path.join(dataset_dir, emotion_name)

    if not os.path.exists(emotion_folder):
        print(f"Error: Folder '{emotion_folder}' does not exist.")
        return

    # Find all image files in the directory
    extensions = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(emotion_folder, ext)))

    if not image_paths:
        print(f"No images found in '{emotion_folder}'.")
        return

    # Sort files so renaming follows original order
    image_paths.sort()

    print(f"Found {len(image_paths)} remaining images in '{emotion_folder}'. Renaming...")

    # Step 1: Rename files to temporary names to prevent filename collision
    temp_paths = []
    for idx, path in enumerate(image_paths):
        ext = os.path.splitext(path)[1]
        temp_name = os.path.join(emotion_folder, f"temp_rename_{idx:04d}{ext}")
        os.rename(path, temp_name)
        temp_paths.append(temp_name)

    # Step 2: Sequentially label back to <emotion>_XXXX.jpg format
    renamed_count = 0
    for idx, temp_path in enumerate(temp_paths):
        ext = os.path.splitext(temp_path)[1]
        new_name = os.path.join(emotion_folder, f"{emotion_name}_{idx:04d}{ext}")
        os.rename(temp_path, new_name)
        renamed_count += 1

    print(f"Successfully relabeled {renamed_count} images from {emotion_name}_0000{ext} to {emotion_name}_{(renamed_count-1):04d}{ext}!")

# --- Usage ---
# Pass the folder name of the emotion dataset you want to clean up:
relabel_dataset_images("thinking_monkey")

# To clean up another emotion folder, call it again:
# relabel_dataset_images("sad")