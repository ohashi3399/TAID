import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# 変換されたモデルパス
model_path = "logs/calm3-taid/epoch=4-step=5.ckpt/hf_model"

# モデルとトークナイザーをロード
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# モデルの説明を準備
model_description = """
# TAID Distilled Model

This model was distilled from Microsoft's calm3 model using the TAID (Temporally Adaptive Interpolated Distillation) approach.

## Model Details
- Teacher model: cyberagent/calm3-22b-chat
- Student model base: TinyLlama/TinyLlama_v1.1
- Distillation method: TAID (Temporally Adaptive Interpolated Distillation)
- Training data: UltraChat 200K
"""

# ハブにアップロード（YOUR_USERNAME/MODEL_NAMEを自分のユーザー名とモデル名に変更）
model.push_to_hub("ryota39/taid-calm3", commit_message="Add TAID distilled model", token=os.environ.get("HUGGINGFACE_API_KEY"))
tokenizer.push_to_hub("ryota39/taid-calm3", commit_message="Add tokenizer for TAID distilled model", token=os.environ.get("HUGGINGFACE_API_KEY"))