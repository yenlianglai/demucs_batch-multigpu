import os
import subprocess
import sys
import torch as th
import torchaudio as ta
from tqdm import tqdm
from io import BytesIO
from .audio import AudioFile, convert_audio

def get_size_from_s3(s3_client, bucket_name, key):
    """Fetch file size in KB from S3 object metadata."""
    try:
        response = s3_client.head_object(Bucket=bucket_name, Key=key)
        return response["ContentLength"] / 1024  # Convert to KB
    except s3_client.exceptions.ClientError as e:
        print(f"Could not access {key} in bucket {bucket_name}: {e}")
        return None

def check_s3_file_exists(s3_client, bucket_name, key):
    """Check if a file exists in an S3 bucket."""
    try:
        s3_client.head_object(Bucket=bucket_name, Key=key)
        return True
    except s3_client.exceptions.ClientError:
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
        self.bucket_name = input_bucket
        self.output_bucket = output_bucket
        self.model_name = model_name
        self.ext = ext
        self.drop_kb = drop_kb
        self.song_ids = song_ids

        self.file_list = self.lazy_file_generator(song_ids)
        print("Number of files available in S3 bucket for processing.")

        self.audio_channels = audio_channels
        self.samplerate = samplerate
        self.audiolength = audiolength

    def lazy_file_generator(self, song_ids):
        """Generator to lazily retrieve file keys from S3 based on song_ids."""
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".aac"):
                    file_id = os.path.splitext(os.path.basename(key))[0]
                    if not song_ids or file_id in song_ids:
                        yield key

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
        size_kb = get_size_from_s3(self.s3_client, self.bucket_name, key)
        if size_kb is None or size_kb < self.drop_kb:
            print(
                f"File {key} is below the size threshold ({self.drop_kb} KB). Skipping."
            )
            return None

        # Load the audio data from S3
        wav = load_track(
            self.s3_client, self.bucket_name, key, self.audio_channels, self.samplerate
        )
        if wav is None:
            raise RuntimeError(f"Failed to load track {key} from S3.")

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
        return sum(1 for _ in self.lazy_file_generator(self.song_ids))

def load_track(s3_client, bucket_name, key, audio_channels, samplerate):
    """Load an audio track from S3."""
    errors = {}
    wav = None

    try:
        # Download the file from S3 to an in-memory BytesIO object
        s3_object = s3_client.get_object(Bucket=bucket_name, Key=key)
        audio_data = BytesIO(s3_object["Body"].read())

        # Use AudioFile if it supports in-memory file data
        wav = AudioFile(audio_data).read(
            streams=0, samplerate=samplerate, channels=audio_channels
        )
    except FileNotFoundError:
        errors["ffmpeg"] = "FFmpeg is not installed."
    except subprocess.CalledProcessError:
        errors["ffmpeg"] = "FFmpeg could not read the file."

    if wav is None:
        try:
            # Read the audio using torchaudio from the in-memory data
            wav, sr = ta.load(audio_data)
        except RuntimeError as err:
            errors["torchaudio"] = err.args[0]
        else:
            wav = convert_audio(wav, sr, samplerate, audio_channels)

    if wav is None:
        print(f"Could not load file {key}. Maybe it is not a supported file format?")
        for backend, error in errors.items():
            print(
                f"When trying to load using {backend}, got the following error: {error}"
            )
        sys.exit(1)
    return wav
