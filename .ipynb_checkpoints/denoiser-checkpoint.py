from scipy.io.wavfile import read, write
import noisereduce as nr
import soundfile as sf
import io
import os
from tqdm import tqdm

root_dir = './data/'

class Denoiser:
    def __init__(self):
        self.data_dir = os.listdir(root_dir)
        self.data_dir.remove('command_labels.csv')
        self.data_dir_len = len(self.data_dir)
        self.denoised_list, self.rate_list = [], []
    
    def get_wav_dir(self, i):
        return os.listdir(root_dir + self.data_dir[i])

    def write_dn_wavs(self, idx):
        print('\nFlushing denoised files to disk ...\n')

        wav_dir = self.get_wav_dir(idx)
        
        # Writing to new file
        for i in tqdm(range(len(self.denoised_list))):
            write(root_dir + self.data_dir[idx] + '/' + wav_dir[i][:-4]+'_dn.wav', self.rate_list[i], self.denoised_list[i])
        
        self.denoised_list, self.rate_list = [], []
    
    def denoise(self):
        print('Starting to denoise sequentially ...')
        for idx1 in tqdm(range(self.data_dir_len)):
            for idx2 in range(len(self.get_wav_dir(idx1))):

                # Individual .wav file
                wav_name = self.get_wav_dir(idx1)[idx2]
                
                # Loading Wav File
                with open(root_dir + self.data_dir[idx1] + '/' + wav_name, "rb") as wavfile:
                    input_wav = wavfile.read()
                
                data, rate = sf.read(io.BytesIO(input_wav))

                # Extracting noise
                noisy_part = data[-20000:]

                # De-noising
                self.denoised_list.append(nr.reduce_noise(audio_clip=data, noise_clip=noisy_part, prop_decrease=1.0))

                # Appending rates
                self.rate_list.append(rate)

            # Writing denoised data
            self.write_dn_wavs(idx1)

        print('Sequential denoising complete.')

dn = Denoiser()

dn.denoise()