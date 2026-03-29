import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy
import argparse

class Display:
    def __init__(self, sr=None) -> None:
        self.audio_files = []
        self.signals = []
        self.sr = sr
    
    def display(self, type=[1, 1, 1], title="", sub_titles=None, begin=0, end=-1, savefig=None, dpi=200):
        # type = [time-domain, frequency-domain, time-frequency-domain]
        for i in range(len(self.audio_files)):
            signal, self.sr = librosa.load(self.audio_files[i], sr=self.sr)
            self.signals.append(signal)
        fig_width = sum(type)
        fig_height = len(self.signals)
        fig = plt.figure(figsize=(5*fig_width, 2*fig_height))
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
        fig.suptitle(title)
        # fig.text(0.03, (0.9+0.9-0.2+0.05)/2, 'y', horizontalalignment='center', fontsize=12, transform=fig.transFigure, fontdict={'fontsize': 12, 'weight': 'bold'})
        # fig.text(0.03, 0.7, 'z', horizontalalignment='center', fontsize=12, transform=fig.transFigure, fontdict={'fontsize': 12, 'weight': 'bold'})
        sub_offset = 0.4/fig.get_size_inches()[1]
        sub_stride = (0.9-0.1) / len(self.signals) 
        sub_top = 0.9
        sub_bottom = (sub_top-sub_stride)
        for i in range(len(self.signals)):
            if sub_titles!=None and i<len(sub_titles):
                fig.text(0.03, (sub_top+sub_bottom)/2.0, sub_titles[i], horizontalalignment='center', fontsize=12, transform=fig.transFigure, fontdict={'fontsize': 12, 'weight': 'bold'})
            sub_top = sub_bottom
            sub_bottom = (sub_top-sub_stride)
            fig_offset = sum(type) * i
            signal = self.signals[i][begin:end]
            # plot time-domain signal
            if type[0]:
                fig_offset += 1
                time = np.linspace(0, 1/self.sr*len(signal), len(signal))
                plt.subplot(fig_height, fig_width, fig_offset)
                plt.xlabel("time(s)")
                plt.ylabel("signal")
                plt.plot(time, signal, linewidth=1)
            # plot frequency-domain
            if type[1]:
                fig_offset += 1
                fre_signal = scipy.fft.fft(signal)
                mag = np.abs(fre_signal)
                phase = np.angle(fre_signal)
                fre = np.linspace(0, self.sr, len(mag))
                plt.subplot(fig_height, fig_width, fig_offset)
                plt.xlabel("frequency(Hz)")
                plt.ylabel("magnitute")
                plt.plot(fre[0:int(len(mag)/2+1)], mag[0:int(len(mag)/2+1)], linewidth=1)
            # plot time-frequency
            if type[2]:
                fig_offset += 1
                FRAME_SIZE = 2048
                HOP_LENGTH = 512
                plt.subplot(fig_height, fig_width, fig_offset)
                spectrum = np.abs(librosa.stft(signal, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH))
                librosa.display.specshow(librosa.amplitude_to_db(spectrum), sr=self.sr, hop_length=HOP_LENGTH, x_axis="time", y_axis="log", cmap="viridis")
                plt.colorbar(format="%+1.f")
            
        
        plt.tight_layout()
        if savefig != None:
            fig.savefig(savefig, dpi=dpi)
        plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="+", help="file paths")
    parser.add_argument("--output", "-o", default="output.png", help="output path")
    args = parser.parse_args()

    output_path = args.output
    files = args.filenames

    display = Display()
    for i in files:
        display.audio_files.append(i)
    
    display.display([1, 1, 1], savefig=output_path)
    plt.show()
