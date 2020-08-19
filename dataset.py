import pandas as pd

import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio.datasets.utils import walk_files


class WavDataset(Dataset):
    """Wav Files Dataset"""

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the audio files.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self._csv = pd.read_csv(csv_file)
        self._csv.sort_values(["wav_dir"], axis=0,
                              ascending=True, inplace=True)
        self._root_dir = root_dir
        walker = walk_files(self._root_dir, suffix='.wav',
                            prefix=True, remove_suffix=False)
        self._walker = sorted(list(walker))
        self._label_list = []

    def __len__(self):
        return len(self._walker)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        for label in range(len(self._csv.iloc[:, 1])):
            for _ in range(len(os.listdir(os.path.dirname(self._walker[0])))):
                self._label_list.append(str(self._csv.iloc[label, 1]))

        wav_name = self._walker[idx]
        wav_label = self._label_list[idx]
        wav_data, sample_rate = torchaudio.load(wav_name)

        return wav_data, wav_label
