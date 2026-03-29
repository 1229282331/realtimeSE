import librosa
import argparse
from evaluation_func import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="+", help="file paths")
    parser.add_argument("--output", "-o", default="output.png", help="output path")
    args = parser.parse_args()

    output_path = args.output
    files = args.filenames
    clean_file = files[0]
    noisy_file = files[1]
    enhanced_file = files[2]
    sr = 16000

    clean, _ = librosa.load(clean_file, sr=sr, dtype="float32")
    noisy, _ = librosa.load(noisy_file, sr=sr, dtype="float32")
    enhanced, _ = librosa.load(enhanced_file, sr=sr, dtype="float32")
    enhanced = enhanced[256:]
    b_SISNR, b_PESQ, b_STOI = eval_all(noisy, clean, sr)
    print("Noisy          > SISNR:{:.4f}, PESQ:{:.4f}, STOI:{:.4f}.".format(b_SISNR, b_PESQ, b_STOI))
    SISNR, PESQ, STOI = eval_all(enhanced, clean[:len(enhanced)], sr)
    print("Enhanced       > SISNR:{:.4f}, PESQ:{:.4f}, STOI:{:.4f}.".format(SISNR, PESQ, STOI))