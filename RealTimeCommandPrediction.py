import numpy as np
import tensorflow as tf
import librosa
import pyaudio
import signal
import sys
import threading
from queue import Queue

SAVED_MODEL_PATH = r"C:\Users\omars\Desktop\New folder\model.keras"
COMMANDS = ["backward", "forward", "go", "left", "right", "stop"]
SILENCE_THRESHOLD = 0.05  # Adjust this threshold based on your needs

# Load the model
model = tf.keras.models.load_model(SAVED_MODEL_PATH)
print("Model loaded successfully!")

def preprocess_audio(audio_signal, sr=22050, n_mfcc=13, n_fft=2048, hop_length=512, duration=1):
    """Preprocess the audio signal to extract MFCCs."""
    # Ensure consistency in length (e.g., trim or pad the audio_signal to the duration)
    max_length = sr * duration
    if len(audio_signal) > max_length:
        audio_signal = audio_signal[:max_length]
    elif len(audio_signal) < max_length:
        pad_width = max_length - len(audio_signal)
        audio_signal = np.pad(audio_signal, (0, pad_width), mode='constant')

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    
    # Transpose to match the expected input shape (time, mfcc)
    mfccs = mfccs.T
    
    # Add an extra dimension to match the input shape expected by the model
    mfccs = mfccs[np.newaxis, ..., np.newaxis]
    
    return mfccs

def predict_command(model, mfccs, commands=COMMANDS):
    """Predict the speech command using the trained model."""
    # Predict using the model
    predictions = model.predict(mfccs)
    
    # Get the index of the highest probability
    predicted_index = np.argmax(predictions, axis=1)[0]
    
    # Map the index to the corresponding command
    predicted_command = commands[predicted_index]
    
    return predicted_command

def signal_handler(sig, frame):
    print('You pressed Ctrl+C! Exiting...')
    sys.exit(0)

def audio_capture(stream, queue, chunk_size):
    while True:
        data = stream.read(chunk_size)
        audio_signal = np.frombuffer(data, dtype=np.float32)
        queue.put(audio_signal)

def main():
    # Set up signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    
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
            # Get the audio signal from the queue
            audio_signal = audio_queue.get()
            
            # Check if the audio signal has significant sound (not silence)
            if np.mean(np.abs(audio_signal)) > SILENCE_THRESHOLD:
                # Preprocess the audio signal
                mfccs = preprocess_audio(audio_signal, sr=RATE)
                
                # Make the prediction
                predicted_command = predict_command(model, mfccs)
                
                # Print the predicted command
                print(f"The predicted command is: {predicted_command}")
    
    except KeyboardInterrupt:
        print("Stopping audio stream...")
    
    finally:
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
