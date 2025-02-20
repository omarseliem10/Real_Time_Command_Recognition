import sounddevice
import numpy as np
import tensorflow as tf
import librosa
import pyaudio
import signal
import sys
import threading
from queue import Queue
import Jetson.GPIO as GPIO

# Constants
SAVED_MODEL_PATH = r"/home/omarseliem10/Desktop/tensorflow/model"
COMMANDS = ["backward", "forward", "go", "left", "right", "stop"]
SILENCE_THRESHOLD = 0.05  # Adjust this threshold based on your needs

# GPIO Pins
RIGHT_PIN = 7
LEFT_PIN = 11
FORWARD_PIN = 12
GO_PIN = 12
BACKWARD_PIN = 13
STOP_PIN = 15

# Load the model
model = tf.keras.models.load_model(SAVED_MODEL_PATH)
print("Model loaded successfully!")

def preprocess_audio(audio_signal, sr=22050, n_mfcc=13, n_fft=2048, hop_length=512, duration=1):
    """Preprocess the audio signal to extract MFCCs."""
    max_length = sr * duration
    if len(audio_signal) > max_length:
        audio_signal = audio_signal[:max_length]
    elif len(audio_signal) < max_length:
        pad_width = max_length - len(audio_signal)
        audio_signal = np.pad(audio_signal, (0, pad_width), mode='constant')

    mfccs = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfccs = mfccs.T
    mfccs = mfccs[np.newaxis, ..., np.newaxis]
    
    return mfccs

def predict_command(model, mfccs, commands=COMMANDS):
    """Predict the speech command using the trained model."""
    predictions = model.predict(mfccs)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_command = commands[predicted_index]
    
    return predicted_command

def signal_handler(sig, frame):
    print('You pressed Ctrl+C! Exiting...')
    GPIO.cleanup()
    sys.exit(0)

def audio_capture(stream, queue, chunk_size):
    while True:
        data = stream.read(chunk_size)
        audio_signal = np.frombuffer(data, dtype=np.float32)
        queue.put(audio_signal)

def setup_gpio():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(RIGHT_PIN, GPIO.OUT)
    GPIO.setup(LEFT_PIN, GPIO.OUT)
    GPIO.setup(FORWARD_PIN, GPIO.OUT)
    GPIO.setup(GO_PIN, GPIO.OUT)
    GPIO.setup(BACKWARD_PIN, GPIO.OUT)
    GPIO.setup(STOP_PIN, GPIO.OUT)

def execute_command(command):
    if command == "backward":
        GPIO.output(BACKWARD_PIN, GPIO.HIGH)
        print("Executing command: backward")
    elif command == "forward" or command == "go":
        GPIO.output(FORWARD_PIN, GPIO.HIGH)
        print("Executing command: forward or go")
    elif command == "left":
        GPIO.output(LEFT_PIN, GPIO.HIGH)
        print("Executing command: left")
    elif command == "right":
        GPIO.output(RIGHT_PIN, GPIO.HIGH)
        print("Executing command: right")
    elif command == "stop":
        GPIO.output(STOP_PIN, GPIO.HIGH)
        print("Executing command: stop")
    else:
        # Reset all pins
        GPIO.output(RIGHT_PIN, GPIO.LOW)
        GPIO.output(LEFT_PIN, GPIO.LOW)
        GPIO.output(FORWARD_PIN, GPIO.LOW)
        GPIO.output(GO_PIN, GPIO.LOW)
        GPIO.output(BACKWARD_PIN, GPIO.LOW)
        GPIO.output(STOP_PIN, GPIO.LOW)

def main():
    # Set up signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    
    # Set up GPIO
    setup_gpio()
    
    # Audio stream parameters
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 22050  # Sample rate
    CHUNK = RATE  # Read in 1-second chunks
    
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    print("Streaming audio... Press Ctrl+C to stop.")
    
    audio_queue = Queue()
    capture_thread = threading.Thread(target=audio_capture, args=(stream, audio_queue, CHUNK))
    capture_thread.daemon = True
    capture_thread.start()
    
    try:
        while True:
            audio_signal = audio_queue.get()
            
            if np.mean(np.abs(audio_signal)) > SILENCE_THRESHOLD:
                mfccs = preprocess_audio(audio_signal, sr=RATE)
                predicted_command = predict_command(model, mfccs)
                print(f"The predicted command is: {predicted_command}")
                execute_command(predicted_command)
    
    except KeyboardInterrupt:
        print("Stopping audio stream...")
    
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        GPIO.cleanup()

if __name__ == "__main__":
    main()
