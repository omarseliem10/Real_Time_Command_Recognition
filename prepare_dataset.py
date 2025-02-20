import librosa
import os
import json
import numpy as np

DATASET_PATH = r"C:\Users\omars\Desktop\New folder"
JSON_PATH = r"C:\Users\omars\Desktop\New folder\data.json"
SAMPLES_TO_CONSIDER = 22050

def preprocess_dataset(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512):
    """Extracts MFCCs from music dataset and saves them into a json file.

    :param dataset_path (str): Path to dataset
    :param json_path (str): Path to json file used to save MFCCs
    :param num_mfcc (int): Number of coefficients to extract
    :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
    :param hop_length (int): Sliding window for FFT. Measured in # of samples
    :return:
    """

    # dictionary where we'll store mapping, labels, MFCCs and filenames
    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    # dictionary to store counts of ignored and processed files
    file_counts = {}

    # loop through all sub-dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're at sub-folder level
        if dirpath != dataset_path:

            # save label (i.e., sub-folder name) in the mapping
            label = os.path.basename(dirpath)
            data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))

            ignored_files_count = 0
            processed_files_count = 0

            # process all audio files in sub-dir and store MFCCs
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # load audio file and slice it to ensure length consistency among different files
                signal, sample_rate = librosa.load(file_path)

                # Remove files with less than the required number of samples
                if len(signal) < SAMPLES_TO_CONSIDER:
                    ignored_files_count += 1
                    continue

                # Ensure consistency of the length of the signal
                signal = signal[:SAMPLES_TO_CONSIDER]

                # Check for silence or mostly silent files
                if np.mean(np.abs(signal)) < 0.01:  # Adjust threshold as needed
                    ignored_files_count += 1
                    continue

                # Normalize the signal
                signal = librosa.util.normalize(signal)

                # Extract MFCCs
                MFCCs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)

                # Store data for analyzed track
                data["MFCCs"].append(MFCCs.T.tolist())
                data["labels"].append(i-1)
                data["files"].append(file_path)
                print("{}: {}".format(file_path, i-1))

                processed_files_count += 1

            # Store the counts for the current folder
            file_counts[label] = {
                "processed": processed_files_count,
                "ignored": ignored_files_count
            }

    # save data in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

    # Print the counts for each folder
    for label, counts in file_counts.items():
        print("\nFolder '{}':".format(label))
        print("Number of processed files: {}".format(counts["processed"]))
        print("Number of ignored files: {}".format(counts["ignored"]))

if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, JSON_PATH)
