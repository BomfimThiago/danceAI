import os
import librosa
import numpy as np
import csv


def extract_audio_features(audio_file_path):
    y, sr = librosa.load(audio_file_path, mono=True)

    # Feature extraction
    length = len(y)
    chroma_stft_mean = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    chroma_stft_var = np.var(librosa.feature.chroma_stft(y=y, sr=sr))
    rms_mean = np.mean(librosa.feature.rms(y=y))
    rms_var = np.var(librosa.feature.rms(y=y))
    spectral_centroid_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_centroid_var = np.var(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth_mean = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_bandwidth_var = np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff_mean = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    rolloff_var = np.var(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zero_crossing_rate_mean = np.mean(librosa.feature.zero_crossing_rate(y))
    zero_crossing_rate_var = np.var(librosa.feature.zero_crossing_rate(y))

    # Extract specific 12 values for mean and variance of MFCCs (mfcc2 to mfcc13)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    specific_mfccs_mean = np.mean(mfccs[0:21, :], axis=1)
    specific_mfccs_var = np.var(mfccs[0:21, :], axis=1)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    harmonic, percussive = librosa.effects.hpss(y)

    # Calcule a média da componente harmônica
    harmony_mean = np.mean(harmonic)

    # Calcule a variância da componente harmônica
    harmony_var = np.var(harmonic)

    # Calcule a média da componente percussiva
    perceptr_mean = np.mean(percussive)

    # Calcule a variância da componente percussiva
    perceptr_var = np.var(percussive)

    # Return the extracted features
    return (
        [
            length,
            chroma_stft_mean,
            chroma_stft_var,
            rms_mean,
            rms_var,
            spectral_centroid_mean,
            spectral_centroid_var,
            spectral_bandwidth_mean,
            spectral_bandwidth_var,
            rolloff_mean,
            rolloff_var,
            zero_crossing_rate_mean,
            zero_crossing_rate_var,
            harmony_mean,
            harmony_var,
            perceptr_mean,
            perceptr_var,
            tempo,
        ]
        + list(specific_mfccs_mean)
        + list(specific_mfccs_var)
        + ["zouk"]
    )


# Function to process a single WAV file
def process_single_wav_file(wav_file_path, csv_output_file):
    extracted_features = extract_audio_features(wav_file_path)

    # Append features to CSV
    with open(csv_output_file, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([os.path.basename(wav_file_path)] + extracted_features)


# Function to process all WAV files in a folder
def process_wav_folder(folder_path, csv_output_file):
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            wav_file_path = os.path.join(folder_path, filename)
            process_single_wav_file(wav_file_path, csv_output_file)


if __name__ == "__main__":
    # Example usage:
    wav_folder_path = "zouk_music/wav"
    csv_output_file = "csv_output.csv"

    header_values = [
        "chroma_stft_mean",
        "chroma_stft_var",
        "rms_mean",
        "rms_var",
        "spectral_centroid_mean",
        "spectral_centroid_var",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_var",
        "rolloff_mean",
        "rolloff_var",
        "zero_crossing_rate_mean",
        "zero_crossing_rate_var",
        "harmony_mean",
        "harmony_var",
        "perceptr_mean",
        "perceptr_var",
        "tempo",
        "mfcc1_mean",
        "mfcc1_var",
        "mfcc2_mean",
        "mfcc2_var",
        "mfcc3_mean",
        "mfcc3_var",
        "mfcc4_mean",
        "mfcc4_var",
        "mfcc5_mean",
        "mfcc5_var",
        "mfcc6_mean",
        "mfcc6_var",
        "mfcc7_mean",
        "mfcc7_var",
        "mfcc8_mean",
        "mfcc8_var",
        "mfcc9_mean",
        "mfcc9_var",
        "mfcc10_mean",
        "mfcc10_var",
        "mfcc11_mean",
        "mfcc11_var",
        "mfcc12_mean",
        "mfcc12_var",
        "mfcc13_mean",
        "mfcc13_var",
        "mfcc14_mean",
        "mfcc14_var",
        "mfcc15_mean",
        "mfcc15_var",
        "mfcc16_mean",
        "mfcc16_var",
        "mfcc17_mean",
        "mfcc17_var",
        "mfcc18_mean",
        "mfcc18_var",
        "mfcc19_mean",
        "mfcc19_var",
        "mfcc20_mean",
        "mfcc20_var",
        "label",
    ]

    # Open CSV file and write header
    with open(csv_output_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            ["filename", "length"] + header_values
        )  # Add your header values

    # Process all WAV files in the folder
    process_wav_folder(wav_folder_path, csv_output_file)
