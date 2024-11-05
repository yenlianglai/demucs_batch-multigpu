# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gc
from io import BytesIO
from pathlib import Path

import boto3
import librosa
import torch as th
import torchaudio as ta
from dora.log import fatal
from torch.utils.data import DataLoader
from tqdm import tqdm

from .apply import BagOfModels
from .apply_multigpu import apply_model
from .audio import save_audio
from .data_utils import DemucsDataSet
from .htdemucs import HTDemucs
from .pretrained import ModelLoadingError, add_model_flags, get_model_from_args

def collate_fn(batch):
    return th.utils.data.dataloader.default_collate(
        [item for item in batch if item is not None]
    )

def save_audio_to_s3(wav_tensor, s3_client, bucket, key, **kwargs):
    """Save an audio Tensor to S3 bucket."""
    buffer = BytesIO()
    ta.save(
        buffer,
        wav_tensor,
        format="wav",
        sample_rate=kwargs["samplerate"],
        bits_per_sample=kwargs["bits_per_sample"],
    )
    buffer.seek(0)
    s3_client.put_object(Bucket=bucket, Key=key, Body=buffer)

def process_and_save_source(
    source, s3_client, bucket, key, samplerate, target_sr=None, **kwargs
):
    """Resample and save a source to S3."""
    if target_sr:
        source = librosa.resample(
            source.detach().cpu().numpy(), orig_sr=samplerate, target_sr=target_sr
        )
    save_audio_to_s3(
        th.Tensor(source),
        s3_client=s3_client,
        bucket=bucket,
        key=key,
        samplerate=samplerate,
        **kwargs,
    )

def get_parser():
    parser = argparse.ArgumentParser(
        "demucs.separate", description="Separate the sources for the given tracks"
    )
    parser.add_argument("input_path", type=Path, help="Path to tracks")
    add_model_flags(parser)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("separated"),
        help="Folder where to put extracted tracks. A subfolder "
        "with the model name will be created.",
    )
    parser.add_argument(
        "--filename",
        default="{track}/{stem}.{ext}",
        help="Set the name of output file. \n"
        'Use "{track}", "{trackext}", "{stem}", "{ext}" to use '
        "variables of track name without extension, track extension, "
        "stem name and default output file extension. \n"
        'Default is "{track}/{stem}.{ext}".',
    )
    parser.add_argument(
        "-c",
        "--clone_subdir",
        default=None,
        help="Cloning sub-directories to the output directory. Get the base Input directory path as input. If None, not cloning.",
    )
    parser.add_argument(
        "-b", "--n_batch", default=1, type=int, help="Batch mode True/False"
    )
    parser.add_argument(
        "-l",
        "--audiolength",
        type=int,
        default=1324800,
        help="Length of the audio(sr) based on model's sr. (44100 based default)",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda" if th.cuda.is_available() else "cpu",
        help="Device to use, default is cuda if available else cpu",
    )
    parser.add_argument(
        "--shifts",
        default=1,
        type=int,
        help="Number of random shifts for equivariant stabilization."
        "Increase separation time but improves quality for Demucs. 10 was used "
        "in the original paper.",
    )
    parser.add_argument(
        "--overlap", default=0.25, type=float, help="Overlap between the splits."
    )
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument(
        "--no-split",
        action="store_false",
        dest="split",
        default=True,
        help="Doesn't split audio in chunks. " "This can use large amounts of memory.",
    )
    split_group.add_argument(
        "--segment",
        type=int,
        help="Set split size of each chunk. "
        "This can help save memory of graphic card. ",
    )
    parser.add_argument(
        "--two-stems",
        dest="stem",
        metavar="STEM",
        default=None,
        help="Only separate audio into {STEM} and no_{STEM}. If 'inst' only no_vocal will be saved.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--int24", action="store_true", help="Save wav output as 24 bits wav."
    )
    group.add_argument(
        "--float32", action="store_true", help="Save wav output as float32 (2x bigger)."
    )
    parser.add_argument(
        "--clip-mode",
        default="rescale",
        choices=["rescale", "clamp"],
        help="Strategy for avoiding clipping: rescaling entire signal "
        "if necessary  (rescale) or hard clipping (clamp).",
    )
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument(
        "--flac", action="store_true", help="Convert the output wavs to flac."
    )
    format_group.add_argument(
        "--mp3", action="store_true", help="Convert the output wavs to mp3."
    )
    parser.add_argument(
        "-sr",
        "--sample_rate",
        type=int,
        default=None,
        help="Output sample rate. Resampleing from 44100Hz",
    )
    parser.add_argument(
        "--mp3-bitrate", default=128, type=int, help="Bitrate of converted mp3."
    )
    parser.add_argument(
        "--mp3-preset",
        choices=range(2, 8),
        type=int,
        default=2,
        help="Encoder preset of MP3, 2 for highest quality, 7 for "
        "fastest speed. Default is 2",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        default=64,
        type=int,
        help="Number of jobs. This can increase memory usage but will "
        "be much faster when multiple cores are available.",
    )
    parser.add_argument(
        "--drop_kb",
        default=180,
        type=int,
        help="Files with size under drop_kb will be omitted, for corrputed file omission.",
    )
    parser.add_argument(
        "--num_worker", default=8, type=int, help="num_worker for DataLoader"
    )

    return parser

def main(opts=None):
    parser = get_parser()
    args = parser.parse_args(opts)

    try:
        model = get_model_from_args(args)
    except ModelLoadingError as error:
        fatal(error.args[0])

    if th.cuda.device_count() > 1:
        model = th.nn.DataParallel(model)
        args.device = "cuda"

    model.cpu()
    model.eval()

    ext = "mp3" if args.mp3 else "flac" if args.flac else "wav"

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        aws_session_token=args.aws_session_token,
        region_name=args.region,
    )

    dataset = DemucsDataSet(
        s3_client=s3_client,
        input_bucket=args.input_bucket,
        audio_channels=model.audio_channels,
        samplerate=model.samplerate,
        output_bucket=args.output_bucket,
        model_name=args.name,
        ext=ext,
        audiolength=args.audiolength,
        drop_kb=args.drop_kb,
        song_ids=(
            set(open(args.song_id_file).read().splitlines())
            if args.song_id_file
            else None
        ),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.n_batch,
        num_workers=args.num_worker,
        collate_fn=collate_fn,
    )

    kwargs = {
        "bitrate": args.mp3_bitrate,
        "preset": args.mp3_preset,
        "clip": args.clip_mode,
        "as_float": args.float32,
        "bits_per_sample": 24 if args.int24 else 16,
    }

    for batch, means, stds, tracks in tqdm(dataloader):
        b_sources = apply_model(
            model,
            batch.to(args.device),
            device=args.device,
            shifts=args.shifts,
            split=args.split,
            overlap=args.overlap,
            progress=True,
            num_workers=args.jobs,
            segment=args.segment,
        )
        for k, sources in enumerate(b_sources):
            sources = (sources * stds[k]) + means[k]
            track_basename = Path(tracks[k]).stem

            if args.stem:
                sources = list(sources)
                s3_key_main = f"{args.filename.format(track=track_basename, stem=args.stem, ext=ext)}"
                source_main = sources.pop(model.sources.index(args.stem))
                process_and_save_source(
                    source_main,
                    s3_client,
                    args.output_bucket,
                    s3_key_main,
                    model.samplerate,
                    args.sample_rate,
                    **kwargs,
                )

                # Saving other sources as a combined "no_<stem>"
                other_stem = sum(sources)
                s3_key_other = f"{args.filename.format(track=track_basename, stem='no_' + args.stem, ext=ext)}"
                process_and_save_source(
                    other_stem,
                    s3_client,
                    args.output_bucket,
                    s3_key_other,
                    model.samplerate,
                    args.sample_rate,
                    **kwargs,
                )
            else:
                for source, name in zip(sources, model.sources):
                    s3_key = f"{args.filename.format(track=track_basename, stem=name, ext=ext)}"
                    process_and_save_source(
                        source,
                        s3_client,
                        args.output_bucket,
                        s3_key,
                        model.samplerate,
                        args.sample_rate,
                        **kwargs,
                    )

            gc.collect()

if __name__ == "__main__":
    main()
