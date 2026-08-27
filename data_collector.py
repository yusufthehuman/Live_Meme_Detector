import cv2
import os
import glob
import urllib.request


def download_cascade():
    xml_filename = "haarcascade_frontalface_default.xml"
    if not os.path.exists(xml_filename):
        print("Downloading face cascade XML file...")
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, xml_filename)
    return xml_filename


def create_dataset_from_droidcam(emotion_name, phone_ip="Your own phone ip adress", output_dir="dataset", target_size=(224, 224)):
    xml_path = download_cascade()
    face_cascade = cv2.CascadeClassifier(xml_path)

    emotion_folder = os.path.join(output_dir, emotion_name)
    os.makedirs(emotion_folder, exist_ok=True)

    # -------------------------------------------------------------
    # Scan existing files and find highest index
    # -------------------------------------------------------------
    existing_files = glob.glob(os.path.join(emotion_folder, f"{emotion_name}_*.jpg"))
    existing_indices = []

    for f in existing_files:
        filename = os.path.basename(f)
        try:
            # Extract number from filename (e.g. "happy_0012.jpg" -> 12)
            num = int(filename.split('_')[1].split('.')[0])
            existing_indices.append(num)
        except (IndexError, ValueError):
            continue

    # Resume counter from (highest index + 1), or start at 0 if folder is empty
    start_index = max(existing_indices) + 1 if existing_indices else 0
    saved_count = start_index

    # Connect to DroidCam
    endpoints = [f"http://{phone_ip}:4747/video", f"http://{phone_ip}:4747/mjpegfeed"]
    cap = None
    for url in endpoints:
        test_cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if test_cap.isOpened():
            ret, frame = test_cap.read()
            if ret and frame is not None and frame.size > 0:
                cap = test_cap
                break
        test_cap.release()

    if cap is None or not cap.isOpened():
        print(f"Error: Could not connect to DroidCam stream at {phone_ip}")
        return

    count = 0
    new_images_added = 0
    print(f"Resuming collection for '{emotion_name}' starting at index {start_index:04d}. Press 'q' to stop.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            continue

        display_frame = frame.copy()

        # Capture every 3rd frame
        if count % 3 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # scaleFactor=1.1 gives tighter face crops with less background noise
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_crop = frame[y:y + h, x:x + w]
                face_resized = cv2.resize(face_crop, target_size)

                filename = os.path.join(emotion_folder, f"{emotion_name}_{saved_count:04d}.jpg")
                cv2.imwrite(filename, face_resized)

                saved_count += 1
                new_images_added += 1

                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Saved: {saved_count}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('DroidCam Dataset Collector', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone! Added {new_images_added} new images. Total images in '{emotion_folder}': {saved_count}")


# --- Run Script ---
create_dataset_from_droidcam(emotion_name="temp", phone_ip="Your own phone ip adress")