"""
Request Camera Permission
Run this first to trigger macOS camera permission dialog
"""

import cv2
import sys

print("=" * 60)
print("Requesting Camera Permission...")
print("=" * 60)
print()
print("This will trigger a permission dialog from macOS.")
print("Please click 'OK' or 'Allow' when prompted.")
print()

try:
    # Try to open camera
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        print("✅ SUCCESS! Camera access granted!")
        print()
        
        # Read one frame to confirm
        ret, frame = cap.read()
        if ret:
            print("✅ Camera is working properly!")
            print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
            print()
            print("You can now run:")
            print("   python accurate_camera_demo.py")
        else:
            print("⚠️  Camera opened but couldn't read frame")
            print("   Try unplugging and replugging your camera")
        
        cap.release()
    else:
        print("❌ FAILED: Camera access denied")
        print()
        print("Please do ONE of the following:")
        print()
        print("Option 1: Reset camera permissions")
        print("   Run in your terminal: tccutil reset Camera")
        print()
        print("Option 2: Grant access manually")
        print("   1. Open System Settings → Privacy & Security → Camera")
        print("   2. Find 'Terminal' or 'Python' in the list")
        print("   3. Toggle it ON")
        print("   4. Restart your terminal")
        print()
        sys.exit(1)
        
except Exception as e:
    print(f"❌ ERROR: {e}")
    print()
    print("Make sure:")
    print("  - Your camera is connected")
    print("  - No other app is using the camera")
    print("  - You have camera permission in System Settings")
    sys.exit(1)

print("=" * 60)



