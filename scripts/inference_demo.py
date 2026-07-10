# -*- coding: utf-8 -*-
"""로컬/서버에서 어댑터 추론 데모. GPU 권장 (bf16), CPU도 동작(느림)."""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER = "newcave/kwater-ailab-4b-lora"   # 또는 로컬 어댑터 폴더 경로

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

QUESTIONS = [
    "「AR6 SSP 기후변화 시나리오 기반 다목적댐 홍수방어기준 개선 연구」에서 사용된 데이터 소스를 나열하시오.",
    "K-water 연구에서 'AR6 SSP scenarios'의 제공 기관은 어디인가?",
]

for q in QUESTIONS:
    msgs = [{"role": "user", "content": q}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                     return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=256, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    print("Q:", q)
    print("A:", tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
    print("-" * 60)
