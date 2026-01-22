import sys
import os

print("--- DEBUG INFO ---")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")

try:
    import mediapipe as mp
    print(f"MediaPipe location: {mp.__file__}")
    print(f"MediaPipe dir content: {dir(mp)}")
except Exception as e:
    print(f"Import failed: {e}")

print("------------------")