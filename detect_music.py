# import librosa.feature as lf
# import librosa.display
# import numpy as np


# def extract_features(file_path):
#     y, sr = librosa.load(file_path, mono=True)
#     mfccs = lf.mfcc(y, sr=sr, n_mfcc=13)
#     spectral_contrast = lf.spectral_contrast(y, sr=sr)
#     chroma = lf.chroma_stft(y, sr=sr)
#     tempo, _ = librosa.beat.beat_track(y, sr)

#     features = np.concatenate(
#         [
#             mfccs.mean(axis=1),
#             spectral_contrast.mean(axis=1),
#             chroma.mean(axis=1),
#             [tempo],
#         ]
#     )

#     return features
import os
import librosa
import pandas as pd


def extract_features(file_path):
    y, sr = librosa.load(file_path, mono=True)
    # Add your feature extraction logic here using Librosa
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Combine features into a list
    features = (
        [file_path]
        + list(mfccs.mean(axis=1))
        + list(spectral_contrast.mean(axis=1))
        + list(chroma.mean(axis=1))
        + [tempo]
    )

    return features


def update_csv(csv_path, wav_directory):
    # Load existing CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # Loop through WAV files in the directory
    for wav_file in os.listdir(wav_directory):
        if wav_file.endswith(".wav"):
            wav_path = os.path.join(wav_directory, wav_file)

            # Extract features for the current WAV file
            features = extract_features(wav_path)

            # Create a dictionary to match the CSV columns
            row_data = {col: val for col, val in zip(df.columns, features)}

            # Append the row to the DataFrame
            df = df.append(row_data, ignore_index=True)

    # Save the updated DataFrame back to the CSV file
    df.to_csv(csv_path, index=False)


# Example usage:
csv_file_path = "path/to/your/csv/file.csv"
wav_directory_path = "path/to/your/wav/files"
update_csv(csv_file_path, wav_directory_path)


if __name__ == "__main__":
    # Example usage:
    csv_file_path = "path/to/your/csv/file.csv"
    wav_directory_path = "path/to/your/wav/files"
    update_csv(csv_file_path, wav_directory_path)
