from pathlib import Path 
import os 
import argparse
import torch as th
from tqdm import tqdm

import demucs.separate, demucs.separate_multigpu

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def get_size(file_path, unit='kb'):
    file_size = os.path.getsize(file_path)
    exponents_map = {'bytes': 0, 'kb': 1, 'mb': 2, 'gb': 3}
    if unit not in exponents_map:
        raise ValueError("Must select from \
        ['bytes', 'kb', 'mb', 'gb']")
    else:
        size = file_size / 1024 ** exponents_map[unit]
        return round(size, 3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path, help = "Input directory folder path")        
    parser.add_argument("-b", "--n_batch", default = 8, help="Batch size")  
    parser.add_argument("-o",
                        "--out",
                        type=Path,
                        default=Path("separated"),
                        help="Folder where to put extracted tracks. A subfolder "
                        "with the model name will be created.")
    parser.add_argument("--filename",
                        default="{track}/{stem}.{ext}",
                        help="Set the name of output file. \n"
                        'Use "{track}", "{trackext}", "{stem}", "{ext}" to use '
                        "variables of track name without extension, track extension, "
                        "stem name and default output file extension. \n"
                        'Default is "{track}/{stem}.{ext}".')
    parser.add_argument("-n",
                        "--model_name",
                        default="mdx_extra",
                        help="Model name")
    parser.add_argument("-l",
                        "--audiolength",
                        type = int,
                        default=1324800,
                        help="Length of the audio(sr) based on model's sr. (44100 based default)")
    parser.add_argument("-d",
                        "--device",
                        default="cuda" if th.cuda.is_available() else "cpu",
                        help="Device to use, default is cuda if available else cpu")
    parser.add_argument("-sr",
                        "--sample_rate",
                        type = int,
                        default=None,
                        help="Output sample rate. Resampleing from 44100Hz")
    parser.add_argument("--shifts",
                        default=1,
                        type=int,
                        help="Number of random shifts for equivariant stabilization."
                        "Increase separation time but improves quality for Demucs. 10 was used "
                        "in the original paper.")
    parser.add_argument("--overlap",
                        default=0.25,
                        type=float,
                        help="Overlap between the splits.")
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument("--no-split",
                             action="store_false",
                             dest="split",
                             default=True,
                             help="Doesn't split audio in chunks. "
                             "This can use large amounts of memory.")
    split_group.add_argument("--segment", type=int,
                             help="Set split size of each chunk. "
                             "This can help save memory of graphic card. ")
    parser.add_argument("--two-stems",
                        dest="stem", metavar="STEM",
                        default = None,
                        help="Only separate audio into {STEM} and no_{STEM}. If 'inst' only no_vocal will be saved.")
    parser.add_argument("--mp3-bitrate", dest = "mp3_bitrate",
                        default=128,
                        type=int,
                        help="Bitrate of converted mp3.")
    parser.add_argument("--mp3-preset", dest = "mp3_preset", choices=range(2, 8), type=int, default=2,
                        help="Encoder preset of MP3, 2 for highest quality, 7 for "
                        "fastest speed. Default is 2")
    parser.add_argument("-j", "--jobs",
                        default=0,
                        type=int,
                        help="Number of jobs. This can increase memory usage but will "
                             "be much faster when multiple cores are available.")
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument("--flac", action="store_true", default=False,
                              help="Convert the output wavs to flac.")
    format_group.add_argument("--mp3", action="store_true", default=False,
                              help="Convert the output wavs to mp3.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--int24", action="store_true",
                       help="Save wav output as 24 bits wav.")
    group.add_argument("--float32", action="store_true",
                       help="Save wav output as float32 (2x bigger).")
    parser.add_argument("--clip-mode", dest = "clip_mode", default="rescale", choices=["rescale", "clamp"],
                        help="Strategy for avoiding clipping: rescaling entire signal "
                             "if necessary  (rescale) or hard clipping (clamp).")
    args = parser.parse_args()
    
    params = ["--two-stems", "inst", "--filename", args.filename,"--clip-mode", args.clip_mode,"-n", args.model_name, "--shifts", str(args.shifts), "--overlap", str(args.overlap), "--batching", "-d", args.device, "-c", str(args.input_path.parent), "--mp3-bitrate", str(args.mp3_bitrate), "--mp3-preset", str(args.mp3_preset), "-j", str(args.jobs), "-o", str(args.out), "-l", str(args.audiolength), "-sr", str(args.sample_rate)]
    

    if args.mp3:
        ext = "mp3"
        params.append("--mp3")
    elif args.flac:
        ext = "flac"
        params.append("--flac")
    else:
        ext = "wav"
    
    if args.int24:
        ext = "mp3"
        params.append("--int24")
    elif args.float32:
        ext = "flac"
        params.append("--float32")
    
    if args.split == False :
        params.append("--no-split")

    if args.stem is not None :
        params.append("--two-stems")
        params.append(args.stem)

    files = list(args.input_path.glob('**/*.*'))
    print("Number of initially loaded files : ", len(files))

    ffiles = []
    for file in tqdm(files):    
        if (args.out / args.model_name / file.parent.relative_to(args.input_path) / (str(file.name.rsplit(".", 1)[0]) + '.' + ext)).exists() == False and get_size(file) > 180:
            ffiles.append(file)
    files = ffiles
    print("Number of files which are not separated yet : ", len(files))

    for b_file in tqdm(batch(files, args.n_batch)):
        params_ = params + [str(file) for file in b_file]
        if th.cuda.device_count() > 1:
            demucs.separate_multigpu.main(params_)
        else:
            demucs.separate.main(params_)

if __name__ == "__main__":
    main()