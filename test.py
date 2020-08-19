import os

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from command_dataset import CommandDataset
from model import SpeechRecognitionModel
from data_processing import (
    cer,
    wer,
    preprocess_model,
    postprocess_model
)

EPOCHS = 100
LEARNING_RATE = 5e-4
BATCH_SIZE = 50
ROOT_DIR = './data/'
CSV_FILE = f'{ROOT_DIR}command_labels.csv'
MODEL_PATH = './command_model.pth'


def test(model, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)

            decoded_preds, decoded_targets = postprocess_model(
                output.transpose(0, 1), labels, label_lengths)

            for j in range(len(decoded_preds)):
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

    avg_cer = sum(test_cer)/len(test_cer)
    avg_wer = sum(test_wer)/len(test_wer)

    print('\nTest set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(
        test_loss, avg_cer, avg_wer))


use_cuda = torch.cuda.is_available()
torch.manual_seed(7)
device = torch.device("cuda" if use_cuda else "cpu")

test_dataset = CommandDataset(csv_file=CSV_FILE, root_dir=ROOT_DIR)

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         collate_fn=lambda x: preprocess_model(x, 'test'),
                         **kwargs)

model = SpeechRecognitionModel().to(device)

if not os.path.isfile(MODEL_PATH):
    RuntimeError("No saved model found. Model test cannot proceed.")

model.load_state_dict(torch.load(MODEL_PATH))

criterion = nn.CTCLoss(blank=28).to(device)

for epoch in tqdm(range(1, EPOCHS + 1)):
    print("\nEvaluating...")
    test(model, device, test_loader, criterion, epoch)
