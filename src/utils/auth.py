import os
import wandb
from huggingface_hub import login as hf_login
from huggingface_hub.utils import RepositoryNotFoundError
from wandb.errors import UsageError

def login():
    """
    環境変数を使用して Hugging Face と W&B にログインする。
    """
    try:
        # 環境変数を取得
        hf_token = os.getenv('HF_TOKEN')
        wandb_api_key = os.getenv('WANDB_API_KEY')
        wandb_project = os.getenv('WANDB_PROJECT')

        # 必要な環境変数が存在するか確認
        if not hf_token:
            raise ValueError("環境変数 'HF_TOKEN' が設定されていません。")
        if not wandb_api_key:
            raise ValueError("環境変数 'WANDB_API_KEY' が設定されていません。")
        if not wandb_project:
            raise ValueError("環境変数 'WANDB_PROJECT' が設定されていません。")

        # Hugging Face にログイン
        try:
            hf_login(token=hf_token)
            print("Hugging Face にログインしました。")
        except (RepositoryNotFoundError) as hf_error:
            raise RuntimeError(f"Hugging Face ログインに失敗しました: {hf_error}")

        # W&B にログイン
        try:
            wandb.login(key=wandb_api_key)
            wandb.init(project=wandb_project)
            print("W&B にログインしました。")
        except UsageError as wandb_error:
            raise RuntimeError(f"W&B ログインに失敗しました: {wandb_error}")

    except ValueError as ve:
        print(f"環境変数エラー: {ve}")
    except RuntimeError as re:
        print(f"ログインエラー: {re}")
    except Exception as e:
        print(f"予期しないエラーが発生しました: {e}")

if __name__ == "__main__":
    login()
