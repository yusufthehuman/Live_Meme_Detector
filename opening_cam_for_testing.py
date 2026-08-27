import cv2

PHONE_IP = "Your own phone ip adress"  # Update this to match your phone screen!

endpoints = [
    f"http://{PHONE_IP}:4747/video",
    f"http://{PHONE_IP}:4747/mjpegfeed"
]

cap = None

for url in endpoints:
    print(f"Testing connection to {url}...")
    test_cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if test_cap.isOpened():
        ret, frame = test_cap.read()
        if ret and frame is not None and frame.size > 0:
            cap = test_cap
            print(f"[SUCCESS] Connected via {url}")
            break
    test_cap.release()

if cap is None or not cap.isOpened():
    print(f"\n[FAIL] Could not connect to DroidCam at IP {PHONE_IP}")
    print("1. Check the Wi-Fi IP on your phone screen.")
    print("2. Make sure the DroidCam desktop app on Windows is CLOSED.")
    print("3. Ensure phone and PC are on the same Wi-Fi network.")
    exit()

print("\nCamera stream active! Press 'q' in the window to close.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to grab frame.")
        break

    cv2.imshow("DroidCam Test Feed", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()