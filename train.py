import os

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from command_dataset import CommandDataset
from model import SpeechRecognitionModel
from data_processing import preprocess_model

EPOCHS = 100
LEARNING_RATE = 5e-4
BATCH_SIZE = 50
ROOT_DIR = './data/'
CSV_FILE = f'{ROOT_DIR}command_labels.csv'
MODEL_PATH = './command_model.pth'


def train(model, device, train_loader, criterion, optimizer, scheduler, epoch):
    model.train()
    data_len = len(train_loader.dataset)
    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)

        loss = criterion(output, labels, input_lengths, label_lengths)
        loss.backward()

        optimizer.step()
        scheduler.step()

        if batch_idx % 1 == 0 or batch_idx == data_len:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(spectrograms), data_len,
                100. * batch_idx / len(train_loader), loss.item()))


use_cuda = torch.cuda.is_available()
torch.manual_seed(7)
device = torch.device("cuda" if use_cuda else "cpu")

train_dataset = CommandDataset(csv_file=CSV_FILE, root_dir=ROOT_DIR)

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          collate_fn=lambda x: preprocess_model(x, 'train'),
                          **kwargs)

model = SpeechRecognitionModel().to(device)

if os.path.isfile(MODEL_PATH):
    print('Saved model checkpoint found, loading it...\n')
    model.load_state_dict(torch.load(MODEL_PATH))

optimizer = optim.AdamW(model.parameters(), LEARNING_RATE)
criterion = nn.CTCLoss(blank=28).to(device)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                          max_lr=LEARNING_RATE,
                                          steps_per_epoch=len(train_loader),
                                          epochs=EPOCHS,
                                          anneal_strategy='linear')


for epoch in tqdm(range(1, EPOCHS + 1)):
    print("\nTraining...")
    train(model, device, train_loader, criterion, optimizer, scheduler, epoch)

torch.save(model.state_dict(), MODEL_PATH)
