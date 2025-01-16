import torch
import torch.nn as nn
import torch.optim as optim
from transformers import logging

# transformersのロガー設定
logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

def train_model(model, train_loader, device, num_epochs=10, lr=0.001):
    """
    モデルを学習させる関数
    Args:
        model: 学習するモデル (CNNClassifier)
        train_loader: トレーニングデータ用のDataLoader
        device: デバイス (CPUまたはCUDA)
        num_epochs: 学習エポック数
        lr: 学習率
    Returns:
        model: 学習済みのモデル
    """
    # 損失関数とオプティマイザの設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # モデルをデバイスに移動
    model.to(device)

    # 学習ループ
    for epoch in range(num_epochs):
        model.train()  # トレーニングモード
        running_loss = 0.0

        # トレーニングデータでの学習
        for waveforms, labels in train_loader:
            waveforms, labels = waveforms.to(device), labels.to(device)

            # 勾配の初期化
            optimizer.zero_grad()

            # 順伝播
            outputs = model(waveforms)
            loss = criterion(outputs, labels)

            # 逆伝播とオプティマイザの更新
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 1エポックの結果をログ
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    return model
