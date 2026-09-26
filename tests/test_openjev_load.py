"""测试 OpenJev 0.8B 加载与 NLI 推理。"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像

from transformers import AutoModelForSequenceClassification, AutoTokenizer

print("加载 tokenizer...")
tok = AutoTokenizer.from_pretrained(
    "AlexWortega/openjev",
    subfolder="qwen3.5-0.8b-nli-v2s-long",
    trust_remote_code=True,
)
print("加载 model...")
model = AutoModelForSequenceClassification.from_pretrained(
    "AlexWortega/openjev",
    subfolder="qwen3.5-0.8b-nli-v2s-long",
    trust_remote_code=True,
)
model.eval()
print("模型加载成功")
print("labels:", model.config.id2label)

# 测试 NLI：premise + hypothesis -> 蕴含/矛盾/中立
premise = "Transformer 模型基于自注意力机制"
hypothesis = "Transformer 使用自注意力"
inputs = tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
import torch
with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
print("probs:", probs.tolist())
print("entailment prob:", float(probs[1]) if "entailment" in str(model.config.id2label).lower() else "unknown")

# 测试矛盾
h2 = "Transformer 完全不使用注意力"
inputs2 = tok(premise, h2, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    logits2 = model(**inputs2).logits
    probs2 = torch.softmax(logits2, dim=-1)[0]
print("矛盾句 probs:", probs2.tolist())

print("[OK] OpenJev 测试通过")
