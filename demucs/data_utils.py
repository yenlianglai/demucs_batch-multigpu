import os
from io import BytesIO

import torch as th
import torchaudio as ta
from botocore.exceptions import ClientError

from .audio import convert_audio

def get_size_from_s3(s3_client, bucket_name, key):
    """Fetch file size in KB from S3 object metadata."""
    try:
        response = s3_client.head_object(Bucket=bucket_name, Key=key)
        return response["ContentLength"] / 1024  # Convert to KB
    except ClientError as e:
        return None

def check_s3_file_exists(s3_client, bucket_name, key):
    """Check if a file exists in an S3 bucket."""
    try:
        s3_client.head_object(Bucket=bucket_name, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return False
        return False

class DemucsDataSet:
    def __init__(
        self,
        s3_client,
        input_bucket,
        audio_channels,
        samplerate,
        output_bucket,
        model_name,
        ext,
        audiolength,
        drop_kb=180,
        song_ids=None,
    ):
        self.s3_client = s3_client
        self.input_bucket = input_bucket
        self.output_bucket = output_bucket
        self.model_name = model_name
        self.ext = ext
        self.drop_kb = drop_kb
        self.song_ids = song_ids

        self.file_list = [self.translate_song_id(song_id) for song_id in song_ids]
        print(f"Number of expected file to processed: {len(self.file_list)}.")

        self.audio_channels = audio_channels
        self.samplerate = samplerate
        self.audiolength = audiolength

    def translate_song_id(self, song_id: str) -> str:
        return f"audio/{song_id}.full.320K.aac"

    def __getitem__(self, idx):
        try:
            key = self.file_list[idx]
        except IndexError:
            raise IndexError(f"Index {idx} out of range for available files.")

        # Check if the processed file already exists in the output bucket
        output_key = f"{self.model_name}/{os.path.splitext(key)[0]}.vocals.{self.ext}"

        if check_s3_file_exists(self.s3_client, self.output_bucket, output_key):
            print(f"File {output_key} already exists in the output bucket. Skipping.")
            return None

        # Check the file size and skip if below threshold
        size_kb = get_size_from_s3(self.s3_client, self.output_bucket, key)
        if size_kb is not None and size_kb > self.drop_kb:
            print(f"File {key} is already separated. Skipping.")
            return None

        # Load the audio data from S3
        wav = load_track(
            self.s3_client, self.input_bucket, key, self.audio_channels, self.samplerate
        )

        if wav is None:
            return None

        # Process the audio data (padding, normalization, etc.)
        if len(wav.shape) == 1:
            th.stack([wav, wav], dim=0)
        if wav.shape[-1] >= self.audiolength:
            wav = wav[:, : self.audiolength]
        else:
            wav = th.cat(
                [wav, th.zeros(wav.shape[0], self.audiolength - wav.shape[1])], dim=-1
            )

        # Add a small epsilon to zero values and normalize the waveform
        is_zero = wav == 0
        wav = wav + is_zero * 1e-7
        ref = wav.mean(0)
        wav -= ref.mean()
        wav /= ref.std()

        return wav, ref.mean(), ref.std(), key

    def __len__(self):
        return len(self.file_list)

def load_track(s3_client, bucket_name, key, audio_channels, samplerate):
    """Load an audio track from S3."""
    errors = {}
    wav = None

    try:
        # Download the file from S3 to an in-memory BytesIO object
        s3_object = s3_client.get_object(Bucket=bucket_name, Key=key)
        audio_data = BytesIO(s3_object["Body"].read())
    except s3_client.exceptions.NoSuchKey:
        errors["S3-client"] = f"File {key} not found in bucket {bucket_name}."
        audio_data = None
    except Exception as e:
        errors["S3-client"] = f"Error retrieving file {key} from S3: {e}"
        audio_data = None

    # Process audio data if successfully loaded
    if audio_data is not None:
        try:
            # Read the audio using torchaudio from the in-memory data
            wav, sr = ta.load(audio_data)
            wav = convert_audio(wav, sr, samplerate, audio_channels)

        except RuntimeError as err:
            errors["torchaudio"] = f"Torchaudio failed to load the file: {err}"
            wav = None
        except Exception as e:
            errors["torchaudio"] = f"Unexpected error in audio processing: {e}"
            wav = None

    # Final error handling and logging
    if wav is None:
        for backend, error in errors.items():
            print(
                f"When trying to load using {backend}, got the following error: {error}"
            )
        return None

    return wav
