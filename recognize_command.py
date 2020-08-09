import os
import time
import math
import struct
import wave
import sys

import pyaudio
import pandas as pd

import torch
import torch.nn.functional as F
import torchaudio

from model import SpeechRecognitionModel
import data_processing as dp

# Logger Import
sys.path.append('..')
from local_server.logger import Logger

# Logger Init
logger = Logger()
logger.connect()

# TODO: Dynamic Threshold
THRESHOLD = 150
FORMAT = pyaudio.paInt16
SHORT_NORMALIZE = (1.0/32768.0)
CHUNK = 1024
CHANNELS = 1
RATE = 16000
S_WIDTH = 2
TIMEOUT_LENGTH = 2
MODEL_PATH = './command_model.pth'
CSV_PATH = './data/command_labels.csv'
WAV_PATH = './temp.wav'
HPARAMS = {
    "n_cnn_layers": 3,
    "n_rnn_layers": 5,
    "rnn_dim": 512,
    "n_class": 29,
    "n_feats": 128,
    "stride": 2,
    "dropout": 0.1,
}


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
                                  rate=RATE,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=CHUNK)

    def record(self):
        logger.log('\nAudio detected, recording now...\n')
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
        logger.log('\nNow Listening...\n')
        while True:
            input = self.stream.read(CHUNK)
            rms_val = self.rms(input)
            if rms_val > THRESHOLD:
                return self.record()

    def write(self, filename, recording):
        global wf
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(recording)
        wf.close()


# Initialising a new model to recognize
model = SpeechRecognitionModel(
    HPARAMS['n_cnn_layers'],
    HPARAMS['n_rnn_layers'],
    HPARAMS['rnn_dim'],
    HPARAMS['n_class'],
    HPARAMS['n_feats'],
    HPARAMS['stride'],
    HPARAMS['dropout'])

# Loading a pre-trained model
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Loading command labels
commands = list(pd.read_csv(CSV_PATH).iloc[:, 1])

while True:

    # Listening and writing to a temp wav file
    l = Listener()
    wav_data = l.listen()
    start_t = time.time()
    l.write(WAV_PATH, wav_data)

    # Reading wav file as a tensor
    waveform, _ = torchaudio.load_wav(WAV_PATH)

    # Data-preprocecessing
    specs = dp.preprocess_reco(waveform)

    # Recognising
    out = model(specs)
    out = F.log_softmax(out, dim=2)
    out = out.transpose(0, 1)

    # Data-postprocessing
    pred = dp.postprocess_reco(out.transpose(0, 1))
    pred_str = ''.join(pred)

    result_dict = {}
    for command in commands:
        result_dict[command] = dp.cer(command, pred_str)

    min_val = min(result_dict.values())

    # Results
    if min_val < 0.5:
        logger.log('Command Recognised -')
        logger.log([key for key in result_dict if result_dict[key] == min_val])
        logger.log('Accuracy Score - {:0.4f}\n'.format(1-min_val))
    elif 0.5 < min_val < 0.7:
        logger.log('Command Partially Recognised -')
        logger.log([key for key in result_dict if result_dict[key] == min_val])
        logger.log('Accuracy Score - {:0.4f}\n'.format(1-min_val))
    else:
        logger.log('Not able to reconize the command. Try again.')

    logger.log("\n--- Total Recognition Time: %.4f seconds ---\n" %
          (time.time() - start_t))
