import torch
from transformers import logging
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.dataset.AudioDataset import AudioDataset
from src.models.cnn import CNNClassifier
from src.train.training_cnn import train_model
from torchaudio.transforms import MelSpectrogram
# transformersのロガー設定
logger = logging.get_logger(__name__)  # モジュール名を指定してロガーを取得
logger.setLevel(logging.INFO)  # ログレベルをINFOに設定

def eval_cnn(model, dataloader, device):
    """
    モデルの評価を行う関数
    Args:
        model: 学習したモデル
        dataloader: DataLoader
        device: 学習に使用したデバイス
    Returns:
        accuracy: モデルの精度
    """
    model.eval()  # 評価モード
    correct = 0
    total = 0
    with torch.no_grad():
        for waveforms, labels in dataloader:
            waveforms, labels = waveforms.to(device), labels.to(device)

            outputs = model(waveforms)
            _, predicted = torch.max(outputs, 1)  # 最大値のインデックスを取得
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logger.info(f"Accuracy: {accuracy:.2f}%")
    return accuracy

def __main__():
    # データセットの準備
    print("now loading dataset")
    transform = MelSpectrogram(sample_rate=16000, n_mels=64)  # メルスペクトログラム変換の設定
    dataset = AudioDataset(transform)  # データセットの準備
    train_size = int(0.8 * len(dataset))  # 学習データの割合
    test_size = len(dataset) - train_size  # テストデータの割合
    torch.manual_seed(42)  # 再現性を確保
    train_dataset, val_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # DataLoaderの作成
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)  # DataLoaderの作成
    print("end loading dataset")

    print("now training model")
    model = CNNClassifier()  # モデルのインスタンス化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # デバイスの設定
    trained_model = train_model(model, train_loader, device, num_epochs=10, lr=0.001)
    print("end training model")
    accuracy = eval_cnn(trained_model, val_loader, device)
    print(f"Accuracy: {accuracy:.2f}%") # 精度の表示]

if __name__ == '__main__':
    __main__()