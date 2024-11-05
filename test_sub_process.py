import logging
import subprocess

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s:%(process)d] %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

def main():

    has_gpu = torch.cuda.is_available()
    logger.info(f"Has GPU Support: {has_gpu}")

    if not has_gpu:
        logger.warning("GPU is recommended for this task!")

    # S3-related parameters

    command = [
        "python3",
        "./separate_from_s3.py",
        "--mp3",
        "-n",
        "htdemucs",
        "-l",
        str(1323000),
        "--num_worker",
        str(10),
        "-sr",
        str(44100),
        "-b",
        str(16),
        "--two-stems",
        "vocals",
        "--drop_kb",
        "300",
        "--song_id_file",
        "./test_song_ids.txt",
        "--filename",
        "{track}.{stem}.{ext}",
        # Add S3-related arguments to the command
        # "--aws_access_key_id",
        # aws_access_key_id,
        # "--aws_secret_access_key",
        # aws_secret_access_key,
        # "--aws_session_token",
        # aws_session_token,
        # "--region",
        # region,
        # "--input_bucket",
        # str(your_input_bucket),
        # "--output_bucket",
        # str(your_output_bucket),
    ]

    logger.info("Starting the separate source process")
    subprocess.run(command, check=True, text=True)

if __name__ == "__main__":
    main()
