# Real-Time Command Recognition

## Overview

This project is a real-time voice command recognition system that uses deep learning to classify spoken commands. The model processes live audio input, extracts relevant features, and predicts commands using a trained neural network. It is built using TensorFlow, Librosa for audio processing, and PyAudio for real-time audio capture.

## Dataset

The dataset used for training comes from the TensorFlow Speech Commands dataset, available on the official [TensorFlow website](https://www.tensorflow.org/datasets/catalog/speech_commands). This dataset contains labeled audio recordings of various spoken commands, which were used to train the model.

## How the Model Works

1. **Data Preprocessing (prepare\_dataset.py)**

   - Loads audio files from the dataset.
   - Extracts Mel-Frequency Cepstral Coefficients (MFCCs) using Librosa.
   - Saves the extracted features and labels in a JSON file.

2. **Model Training (train.py)**

   - Loads the preprocessed dataset.
   - Splits the data into training, validation, and test sets.
   - Defines a Convolutional Neural Network (CNN) model with multiple layers for feature extraction and classification.
   - Trains the model using TensorFlow/Keras.
   - Saves the trained model for future inference.

3. **Real-Time Prediction (RealTimeCommandPrediction.py)**

   - Captures live audio using PyAudio.
   - Preprocesses the audio input and extracts MFCC features.
   - Loads the trained model and predicts the command in real-time.
   - Outputs the recognized command.

4. **Jetson Nano Version**

   - A modified version of the real-time command recognition script has been uploaded for compatibility with the NVIDIA Jetson Nano.
   - This version optimizes performance and hardware compatibility for edge AI applications.
   - The Jetson Nano version includes GPIO pin control for executing recognized commands on connected hardware. The script `RealTimeCommandRecognition_JetsonNano.py` integrates GPIO handling and optimized real-time processing.

## Installation

### Requirements

Ensure you have Python installed, then install the required dependencies:

```sh
pip install -r requirements.txt
```

### Clone the Repository

```sh
git clone https://github.com/YOUR_USERNAME/real-time-command-recognition.git
cd real-time-command-recognition
```

## Usage

### Step 1: Prepare the Dataset

Run the following command to preprocess the dataset and extract MFCC features:

```sh
python prepare_dataset.py
```

### Step 2: Train the Model

Train the CNN model with the extracted features:

```sh
python train.py
```

### Step 3: Run Real-Time Command Recognition

Start real-time voice command recognition:

```sh
python RealTimeCommandPrediction.py
```

For Jetson Nano users, run the modified version:

```sh
python RealTimeCommandRecognition_JetsonNano.py
```

## Model Architecture

The model is a Convolutional Neural Network (CNN) designed for audio classification. It consists of:

- Three convolutional layers with batch normalization and dropout.
- A fully connected dense layer for classification.
- A softmax output layer to classify audio into predefined commands.

## Commands Recognized

The model is trained to recognize the following spoken commands:

- `backward`
- `forward`
- `go`
- `left`
- `right`
- `stop`

## Acknowledgments

- Dataset: [TensorFlow Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands)
- Frameworks: TensorFlow, Librosa, PyAudio
- Inspired by various speech recognition projects and deep learning research.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

