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


EPOCHS = 25
MAX_LEARNING_RATE = 5e-4
BATCH_SIZE = 10
ROOT_DIR = './data/'
CSV_FILE = f'{ROOT_DIR}command_labels.csv'
MODEL_PATH = './command_model.pth'
CHECKPOINT_PATH = './command_model_chkpnt.pth'
MIN_LOSS = 10


def train(model, device, train_loader, criterion, optimizer, scheduler, epoch):
    global MIN_LOSS
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
            print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(spectrograms), data_len,
                100. * batch_idx / len(train_loader), loss.item()))
            
            if loss.item() < MIN_LOSS:
                # Saving model checkpoint
                print(f'Saving model checkpoint with min loss of {loss.item()}...')
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                MIN_LOSS = loss.item()


use_cuda = torch.cuda.is_available()
torch.manual_seed(7)
#device = torch.device('cuda' if use_cuda else 'cpu')
device = 'cpu'
print(f'Using {device}')

train_dataset = CommandDataset(csv_file=CSV_FILE, root_dir=ROOT_DIR)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          collate_fn=lambda x: preprocess_model(x, 'train'))

model = SpeechRecognitionModel().to(device)

if os.path.isfile(MODEL_PATH):
    print('Saved model checkpoint found, loading it...')
    model.load_state_dict(torch.load(MODEL_PATH))

optimizer = optim.AdamW(model.parameters(), MAX_LEARNING_RATE)
criterion = nn.CTCLoss(blank=28).to(device)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                          max_lr=MAX_LEARNING_RATE,
                                          steps_per_epoch=len(train_loader),
                                          epochs=EPOCHS,
                                          anneal_strategy='linear')

print('Training...')

for epoch in tqdm(range(1, EPOCHS + 1)):
    train(model, device, train_loader, criterion, optimizer, scheduler, epoch)

print(f'Saving trained model with the least loss of {MIN_LOSS}...')
model.load_state_dict(torch.load(CHECKPOINT_PATH))
torch.save(model.state_dict(), 'command_model_trained.pth')
