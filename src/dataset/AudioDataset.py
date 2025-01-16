import os
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class AudioDataset(Dataset):
    def __init__(self, transform=None, max_length=16000):
        self.files = []
        self.labels = []
        self.transform = transform
        self.max_length = max_length  # 最大の波形長さを指定

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        kanata_dir = os.path.join(base_dir, "data/kanata")
        expanded_dir = os.path.join(base_dir, "data/expanded")
        for idx,file in enumerate(os.listdir(kanata_dir)):
            if file.endswith('.wav'):
                self.files.append(os.path.join(kanata_dir, file))
                self.labels.append(1)
            if idx > 100:
                break

        for idx,file in enumerate(os.listdir(expanded_dir)):
            if file.endswith('.wav'):
                self.files.append(os.path.join(expanded_dir, file))
                self.labels.append(0)
            if idx > 100:
                break

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]

        # 音声データの読み込み
        waveform, sample_rate = torchaudio.load(file_path)
        # 波形を一定の長さにパディング
        if waveform.size(1) < self.max_length:
            waveform = F.pad(waveform, (0, self.max_length - waveform.size(1)))
        else:
            waveform = waveform[:, :self.max_length]
        if self.transform is not None:
            waveform = self.transform(waveform)
        return waveform, label
    
def __main__():
    # メルスペクトログラム変換の設定
    transform = MelSpectrogram(sample_rate=16000, n_mels=64)

    # データセットの準備
    dataset = AudioDataset(transform=transform)

    # データローダーの準備
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # ミニバッチの取得
    for waveform, label in dataloader:
        print(waveform.size(), label.size())

if __name__ == '__main__':
    __main__()