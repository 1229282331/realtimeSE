from pesq import pesq
import librosa
import numpy as np
from pystoi import stoi

def cal_speed(sr=16000, nfft=512, hop_len=256):
    frames_num = (sr-nfft)/hop_len + 1
    per_cost = 1 / frames_num * 1000  # 算法每帧处理的时间（ms）

    return per_cost

def cal_psd(X, nfft):
    psd = np.abs(X)**2 / nfft
    return psd
def psd2mag(psd, nfft):
    mag = np.sqrt(psd * nfft)
    return mag

def cal_pesq(clean, enhanced, sr, mode="wb"):
    single_pesq = pesq(sr, clean, enhanced, mode)
    return single_pesq

def cal_stoi(clean, enhanced, sr, extended=False):
    STOI = stoi(clean, enhanced, sr, extended=extended)

def cal_snr(clean_signal, noise_signal):
    clean_power = np.mean(clean_signal**2)
    noise_power = np.mean(noise_signal**2)
    return 10.0 * np.log10(clean_power/(noise_power+1e-12))

def sisnr(esti, tagt):
    """ for single wav """
    esti = esti - esti.mean()
    tagt = tagt - tagt.mean()

    a = np.sum(esti * tagt) / np.sum(tagt**2 + 1e-8)
    e_tagt = a * tagt
    e_res = esti - e_tagt

    return 10*np.log10((np.sum(e_tagt**2)+1e-8) / (np.sum(e_res**2)+1e-8))

def eval_all(esti, tagt, sr=16000):
    SISNR = sisnr(esti, tagt)
    PESQ = cal_pesq(tagt, esti, sr, "wb")
    STOI = cal_stoi(tagt, esti, sr, True)
    if STOI == None:
        STOI = 0.0
    return SISNR, PESQ, STOI

if __name__ == "__main__":
    # noisy_file = "C:/vscode_files/python_files/SEtrain-pro/datasets/test_small/noisy/SI1573_white_snr0_52.wav"
    # noisy_file = "C:/vscode_files/python_files/SEtrain-pro/datasets/test_small/noisy/SI1573_white_snr5_53.wav"
    # noisy_file = "C:/vscode_files/python_files/SEtrain-pro/datasets/test_small/noisy/SI1573_white_snr10_54.wav"
    noisy_file = "C:/vscode_files/python_files/SEtrain-pro/datasets/test_small/noisy/SI1573_myself_noise_snr0_22.wav"
    # noisy_file = "C:/vscode_files/python_files/SEtrain-pro/datasets/test_small/noisy/SI1573_myself_noise_snr5_23.wav"
    # noisy_file = "C:/vscode_files/python_files/SEtrain-pro/datasets/test_small/noisy/SI1573_myself_noise_snr10_24.wav"
    # noisy_file = "C:/vscode_files/python_files/SEtrain-pro/datasets/test_small/noisy/SI1573_babble_snr0_7.wav"
    # noisy_file = "C:/vscode_files/python_files/SEtrain-pro/datasets/test_small/noisy/SI1573_babble_snr5_8.wav"
    # noisy_file = "C:/vscode_files/python_files/SEtrain-pro/datasets/test_small/noisy/SI1573_babble_snr10_9.wav"
    clean_file = "C:/vscode_files/python_files/SEtrain-pro/datasets/TIMIT/data/TEST/DR1/FAKS0/SI1573.WAV.wav"

    noisy, sr = librosa.load(noisy_file, sr=None)
    clean, sr = librosa.load(clean_file, sr=sr)
    print(eval_all(noisy, clean, sr=sr))

    print(cal_snr(clean, noisy-clean))