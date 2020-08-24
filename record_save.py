import pyaudio
import math
import struct
import wave
import time
import os
import sys
from argparse import ArgumentParser
from configparser import ConfigParser

import pandas as pd


config = ConfigParser()
config.read('../audio_processing/audio_config.ini')

FORMAT = getattr(pyaudio, config.get('config', 'FORMAT'))
THRESHOLD = config.getint('config', 'THRESHOLD')
CHUNK = config.getint('config', 'CHUNK_SIZE')
CHANNELS = config.getint('config', 'CHANNELS')
S_RATE = config.getint('config', 'SAMPLE_RATE')
S_WIDTH = config.getint('config', 'SAMPLE_WIDTH')
SHORT_NORMALIZE = (1.0/32768.0)
TIMEOUT_LENGTH = 2


class Recorder:

    @staticmethod
    def rms(frame):
        count = len(frame) / S_WIDTH
        format = "%dh" % (count)
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * SHORT_NORMALIZE
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)

        return rms * 1000

    @staticmethod
    def get_file_dir(self, idx, i):
        # Extracting first letter from each word of dir
        words = self.command_dirs[idx].split('_')
        letters = [word[0] for word in words]

        if not os.path.exists(self.root_dir + self.command_dirs[idx]):
            os.makedirs(self.root_dir + self.command_dirs[idx])

        self.filename = self.root_dir + \
            self.command_dirs[idx] + ''.join(letters) + '-{}.wav'.format(i+1)

    def __init__(self, root_dir, csv_file, samples):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=S_RATE,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=CHUNK)
        self.root_dir = root_dir
        self.samples = samples
        self.command_csv = pd.read_csv(csv_file)
        self.command_dirs = self.command_csv.iloc[:, 0]
        self.command_labels = self.command_csv.iloc[:, 1]

    def record(self):
        print('Audio detected, recording now ...')
        rec = []
        current = time.time()
        end = time.time() + TIMEOUT_LENGTH

        while current <= end:
            data = self.stream.read(CHUNK)
            if self.rms(data) >= THRESHOLD:
                end = time.time() + TIMEOUT_LENGTH

            current = time.time()
            rec.append(data)
        self.write(b''.join(rec))

    def write(self, recording):
        print('Saving audio sample ...')
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(S_RATE)
        wf.writeframes(recording)
        wf.close()
        print('Written to file: {}'.format(self.filename))
        print('Listening again ...')

    def listen(self):
        for idx in range(len(self.command_dirs)):
            print('Listening for \"{}\" ...\n'.format(
                self.command_labels[idx]))
            for i in range(self.samples):
                while True:
                    input = self.stream.read(CHUNK)
                    rms_val = self.rms(input)
                    if rms_val > THRESHOLD:
                        self.get_file_dir(self, idx, i)
                        self.record()
                        break
        print('Done.')


parser = ArgumentParser(
    description='Records and saves a stream of audio as wavfile from default input device inside a root directory according to a csv file')

parser.add_argument('-r', '--root', dest='root', type=str, required=False,
                    default='./data/', help='Root Directory to store the audio files')
parser.add_argument('-c', '--csv', dest='csv', type=str, required=False,
                    default='./data/command_labels.csv', help='CSV file contaning the sentences')
parser.add_argument('-s', '--samples', dest='samples', type=int, required=False,
                    default=5, help='Number of audio samples to take per sentence')

args = parser.parse_args()

a = Recorder(args.root, args.csv, args.samples)

a.listen()
