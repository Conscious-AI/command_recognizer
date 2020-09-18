import os
import time
import math
import struct
import wave
import sys

# TODO: Refactor this spaghetti
PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append('..')
sys.path.append(PATH)

import pyaudio
import pandas as pd

import torch
import torch.nn.functional as F
import torchaudio

from model import SpeechRecognitionModel
import data_processing as dp

# Logger Import
from local_server.logger import Logger
# Audio configs vars
from audio_processing.audio_confs import *


SHORT_NORMALIZE = (1.0/32768.0)
TIMEOUT_LENGTH = 2
MODEL_PATH = f'{PATH}\\command_model_trained.pth'
CSV_PATH = f'{PATH}\\data\\command_labels.csv'
WAV_PATH = f'{PATH}\\temp.wav'

# Logger Init
logger = Logger()
try:
    logger.connect()
    logger_connected = True
except Exception:
    # print('An exception occured while trying to connect to logger')
    logger_connected = False


def _print_log(data):
    # print(data)
    if logger_connected:
        logger.log(f'RECO: {data}')


class Listener:

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

    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=S_RATE,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=CHUNK)

    def record(self):
        _print_log('\nAudio detected, recording now...\n')
        rec = []
        current = time.time()
        end = time.time() + TIMEOUT_LENGTH

        while current <= end:
            data = self.stream.read(CHUNK)
            if self.rms(data) >= THRESHOLD:
                end = time.time() + TIMEOUT_LENGTH

            current = time.time()
            rec.append(data)
        return b''.join(rec)

    def listen(self):
        _print_log('\nNow Listening...\n')
        while True:
            input = self.stream.read(CHUNK)
            rms_val = self.rms(input)
            if rms_val > THRESHOLD:
                return self.record()

    def write(self, filename, recording):
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(S_RATE)
        wf.writeframes(recording)
        wf.close()


class CommandRecognizer():
    def __init__(self):
        # Initialising a new model to recognize
        self.model = SpeechRecognitionModel()

        # Loading a pre-trained model
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.eval()

        # Loading command labels
        self.commands = list(pd.read_csv(CSV_PATH).iloc[:, 1])

    def recognize(self):
        # Listening and writing to a temp wav file
        l = Listener()
        wav_data = l.listen()
        start_t = time.time()
        l.write(WAV_PATH, wav_data)

        # Reading wav file as a tensor
        # TODO: Read directly from the recorded frames
        waveform, _ = torchaudio.load_wav(WAV_PATH)

        # Data-preprocecessing
        specs = dp.preprocess_reco(waveform)

        # Recognising
        out = self.model(specs)
        out = F.log_softmax(out, dim=2)
        out = out.transpose(0, 1)

        # Data-postprocessing
        pred = dp.postprocess_reco(out.transpose(0, 1))
        pred_str = ''.join(pred)

        result_dict = {}
        for command in self.commands:
            result_dict[command] = dp.cer(command, pred_str)

        min_cer = min(result_dict.values())

        # Results
        accuracy = 1 - min_cer
        result = ''.join([key for key in result_dict if result_dict[key] == min_cer])

        if min_cer < 0.5:
            _print_log('Command Recognised -')
            _print_log(result)
            _print_log('Accuracy Score - {:0.4f}\n'.format(accuracy))

        elif 0.5 < min_cer < 0.7:
            _print_log('Command Partially Recognised -')
            _print_log(result)
            _print_log('Accuracy Score - {:0.4f}\n'.format(accuracy))

        else:
            _print_log('Not able to recognize the command. Try again.')
            return None, accuracy

        _print_log("\n--- Total Recognition Time: {:0.4f} seconds ---\n".format(time.time() - start_t))
        return result, accuracy


if __name__ == "__main__":
    recognizer = CommandRecognizer()
    while True:
        recognizer.recognize()