import os
import torchaudio
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from transformers import AutoFeatureExtractor, TrainingArguments, Trainer
from transformers import AutoModelForAudioClassification
from datasets import DatasetDict
from transformers import DataCollatorWithPadding
import torch.nn.functional as F
import random
from transformers import AutoModel

class HuBERTWithLogMel(nn.Module):
    def __init__(self, hubert_model, num_labels=2):
        super(HuBERTWithLogMel, self).__init__()
        self.hubert = hubert_model
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_mel = nn.Linear(1228800, hubert_model.config.hidden_size)
        self.classifier = nn.Linear(hubert_model.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_values, attention_mask, mel_spec, labels=None):
        # HuBERTの出力
        hubert_output = self.hubert(input_values=input_values, attention_mask=attention_mask)
        hubert_hidden_state = hubert_output.last_hidden_state[:, 0, :]  # [CLS]トークンの出力

        # CNNでログメルスペクトログラムを処理
        cnn_output = self.cnn(mel_spec)
        print("Before CNN:", mel_spec.shape)  # CNN入力形状を確認
        cnn_output = cnn_output.view(cnn_output.size(0), -1)  # フラット化
        print("After CNN:", cnn_output.shape)  # CNN出力形状を確認
        mel_hidden_state = self.fc_mel(cnn_output)  # HuBERT隠れ層次元に合わせる

        # HuBERTの出力とログメルスペクトログラムの出力を加算
        combined_hidden_state = hubert_hidden_state + mel_hidden_state

        # 分類層
        logits = self.classifier(combined_hidden_state)

        # ラベルが指定されている場合、損失を計算
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        # 損失とlogitsを返す
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


# データセットの準備関数
def load_audio_data(feature_extractor, expanded_dir, kanata_dir, max_length, sampling_rate, mel_transform):
    data = []

    for label, directory in enumerate([expanded_dir, kanata_dir]):
        for idx, file in enumerate(os.listdir(directory)):
            if file.endswith('.wav') and idx < 3:  # 制限をかけてサンプル数を減らす
                filepath = os.path.join(directory, file)
                waveform, sr = torchaudio.load(filepath)

                # Resample
                if sr != sampling_rate:
                    resampler = torchaudio.transforms.Resample(sr, sampling_rate)
                    waveform = resampler(waveform)

                # パディングまたは切り取り
                if waveform.size(1) < max_length:
                    waveform = F.pad(waveform, (0, max_length - waveform.size(1)))
                else:
                    waveform = waveform[:, :max_length]

                # 特徴量抽出
                inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=sampling_rate, return_attention_mask=True)
                mel_spec = mel_transform(waveform)  # ログメルスペクトログラムの抽出
                mel_spec = torch.log1p(mel_spec)  # 対数を取る

                data.append({
                    "label": label,
                    "input_values": inputs["input_values"][0],
                    "attention_mask": inputs["attention_mask"][0],
                    "mel_spec": mel_spec
                })

    return data

def prepare_dataset(feature_extractor, expanded_dir, kanata_dir, max_length, sampling_rate, mel_transform, train_split=0.8):
    data = load_audio_data(feature_extractor, expanded_dir, kanata_dir, max_length, sampling_rate, mel_transform)

    # シャッフル
    random.shuffle(data)

    # データ分割
    train_size = int(len(data) * train_split)
    train_data = data[:train_size]
    test_data = data[train_size:]

    # DatasetDict の作成
    def convert_to_dict(data):
        return {
            "label": [item["label"] for item in data],
            "input_values": [item["input_values"] for item in data],
            "attention_mask": [item["attention_mask"] for item in data],
            "mel_spec": [item["mel_spec"] for item in data],
        }

    dataset = DatasetDict({
        "train": Dataset.from_dict(convert_to_dict(train_data)),
        "test": Dataset.from_dict(convert_to_dict(test_data)),
    })
    return dataset

# フォルダパスを指定
expanded_dir = "/root/workspace/data/expanded"
kanata_dir = "/root/workspace/data/kanata"

# Feature Extractor のロード
feature_extractor = AutoFeatureExtractor.from_pretrained('rinna/japanese-hubert-base')

# サンプリングレートと最大長
sampling_rate = feature_extractor.sampling_rate
max_length = int(sampling_rate * 30)  # 30秒

# ログメルスペクトログラム変換
mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sampling_rate, n_mels=128)

# データセットの準備
print("Preparing dataset...")
dataset = prepare_dataset(feature_extractor, expanded_dir, kanata_dir, max_length, sampling_rate, mel_transform)
print("Done.")

# HuBERTモデルのロード（隠れ層出力を取得可能なモデル）
hubert_model = AutoModel.from_pretrained('rinna/japanese-hubert-base')

# カスタムモデルの初期化
model = HuBERTWithLogMel(hubert_model, num_labels=2)

# トレーニング設定
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    report_to="wandb",
    run_name="audio-classification"
)

# Data Collator
data_collator = DataCollatorWithPadding(tokenizer=feature_extractor)

# Trainer の初期化
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=feature_extractor,
    data_collator=data_collator
)

# トレーニング開始
trainer.train()
