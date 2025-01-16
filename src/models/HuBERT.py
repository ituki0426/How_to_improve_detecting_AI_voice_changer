import os
import torchaudio
from datasets import Dataset, DatasetDict
from transformers import AutoFeatureExtractor
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
from datasets import DatasetDict
import wandb
from transformers import DataCollatorWithPadding
import torch.nn.functional as F  # torch.nn.functional をインポート

def load_audio_data(feature_extractor, expanded_dir, kanata_dir, max_length, sampling_rate):
    # データ格納用リスト
    data = []

    # Label 1 のデータを読み込む
    for idx,file in enumerate(os.listdir(expanded_dir)):
        if file.endswith('.wav') and idx < 5:
            filepath = os.path.join(expanded_dir, file)
            waveform, sr = torchaudio.load(filepath)

            # Resample する場合（サンプリングレートが異なる場合）
            if sr != sampling_rate:
                resampler = torchaudio.transforms.Resample(sr, sampling_rate)
                waveform = resampler(waveform)

            # パディング処理の修正
            if waveform.size(1) < max_length:
                waveform = F.pad(waveform, (0, max_length - waveform.size(1)))
            else:
                waveform = waveform[:, :max_length]

            # 特徴量抽出
            inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=sampling_rate, return_attention_mask=True)
            data.append({
                "label": 0,
                "input_values": inputs["input_values"][0],
                "attention_mask": inputs["attention_mask"][0]
            })

    # Label 2 のデータを読み込む
    for idx,file in enumerate(os.listdir(kanata_dir)):
        if file.endswith('.wav') and idx < 5:
            filepath = os.path.join(kanata_dir, file)
            waveform, sr = torchaudio.load(filepath)

            # Resample する場合
            if sr != sampling_rate:
                resampler = torchaudio.transforms.Resample(sr, sampling_rate)
                waveform = resampler(waveform)

            # パディングまたは切り取りで長さを統一
            if waveform.size(1) < max_length:
                waveform = F.pad(waveform, (0, max_length - waveform.size(1)))
            else:
                waveform = waveform[:, :max_length]

            # 特徴量抽出
            inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=sampling_rate, return_attention_mask=True)
            data.append({
                "label": 1,
                "input_values": inputs["input_values"][0],
                "attention_mask": inputs["attention_mask"][0]
            })

    return data

def prepare_dataset(feature_extractor, expanded_dir, kanata_dir, max_length, sampling_rate, train_split=0.8):
    # データをロード
    data = load_audio_data(feature_extractor, expanded_dir, kanata_dir, max_length, sampling_rate)

    # シャッフル
    import random
    random.shuffle(data)

    # データ分割
    train_size = int(len(data) * train_split)
    train_data = data[:train_size]
    test_data = data[train_size:]

    # リストを辞書形式に変換する関数
    def convert_to_dict(data):
        return {
            "label": [item["label"] for item in data],
            "input_values": [item["input_values"] for item in data],
            "attention_mask": [item["attention_mask"] for item in data],
        }

    # DatasetDict の作成
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

# データセットの準備
print("Preparing dataset...")
dataset = prepare_dataset(feature_extractor, expanded_dir, kanata_dir, max_length, sampling_rate)
print("Done.")
# 結果を確認
print(dataset)

model = AutoModelForAudioClassification.from_pretrained(
    'rinna/japanese-hubert-base',
    num_labels=2,  # ラベル数 (1 と 2 の2種類)
    id2label={0: "label_1", 1: "label_2"},
    label2id={"label_1": 0, "label_2": 1}
)

training_args = TrainingArguments(
    output_dir="./results",  # モデル保存先
    evaluation_strategy="epoch",  # 各エポック終了時に評価
    save_strategy="epoch",  # 各エポック終了時にモデル保存
    learning_rate=5e-5,
    per_device_train_batch_size=4,  # バッチサイズ
    per_device_eval_batch_size=4,
    num_train_epochs=10,  # エポック数
    warmup_steps=500,  # ウォームアップステップ
    weight_decay=0.01,
    logging_dir="./logs",  # ログの保存先
    logging_steps=10,
    save_total_limit=2,  # 保存モデル数の上限
    report_to="wandb",  # W&B を使う場合
    run_name="audio-classification"
)

data_collator = DataCollatorWithPadding(tokenizer=feature_extractor)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],  # トレーニングデータセット
    eval_dataset=dataset["test"],   # テストデータセット
    tokenizer=feature_extractor,
    data_collator=data_collator
)

# トレーニング開始
trainer.train()