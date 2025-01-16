import torch.nn as nn
import torch.nn.functional as F
import torch

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # 畳み込み層
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # バッチ正規化
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # バッチ正規化

        # プーリング層
        self.pool = nn.MaxPool2d(2, 2)  # 2x2プーリング

        # 全結合層の入力サイズを計算
        self.fc_input_size = 32 * (64 // 4) * (81 // 4)

        # 全結合層
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.dropout = nn.Dropout(0.5)  # ドロップアウト
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # 畳み込み -> バッチ正規化 -> ReLU -> プーリング
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [B, 16, 32, 40]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [B, 32, 16, 20]

        # フラット化
        x = x.view(-1, self.fc_input_size)

        # 全結合層 -> ドロップアウト -> 出力
        x = F.relu(self.fc1(x))  # [B, 128]
        x = self.dropout(x)      # ドロップアウト
        x = self.fc2(x)          # [B, 2]
        return x


def __main__():
    model = CNNClassifier()
    batch_size = 32
    dummy_input = torch.randn(batch_size, 1, 64, 81)  # メルスペクトログラムの形状
    model = CNNClassifier()
    # フォワードパス
    output = model(dummy_input)
    print("Output shape:", output.shape)  # 期待される形状: [32, 2]
    print(model)

if __name__ == '__main__':
    __main__()