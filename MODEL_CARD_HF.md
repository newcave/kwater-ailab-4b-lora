---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
language:
- ko
- en
tags:
- lora
- peft
- korean
- water-resources
- k-water
library_name: peft
pipeline_tag: text-generation
---

<p align="center"><img src="AI_Lab_logo.jpg" width="200"/></p>

# kwater-ailab-4b-lora

K-water AI연구소(K-water AI Research Institute)의 첫 LoRA 파인튜닝 언어모델입니다.
K-water 연구보고서 105건에서 추출한 데이터소스 언급(datasource mentions) 구조화 데이터로 학습하여,
연구 데이터소스에 관한 질문에 출처를 표기하며 답하도록 조정되었습니다.

The first LoRA fine-tuned language model from K-water AI Research Institute (Republic of Korea),
trained on structured datasource-mention records extracted from 105 K-water research reports.
A proof-of-concept for the **Water Co-Scientist** research-agent initiative.

## Details

- **Base**: Qwen/Qwen3-4B-Instruct-2507 (Apache 2.0)
- **Method**: LoRA r=16 / alpha=32, all linear layers, bf16, 2 epochs
- **Data**: rule-generated instruction pairs (listing / field lookup / summary) from datasource-mention records
- **Hardware**: NVIDIA A100 40GB (Google Colab)

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3-4B-Instruct-2507"
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(model, "newcave/kwater-ailab-4b-lora")
```

## Limitations

v0.1 proof-of-concept. Trained on template-generated pairs; primarily learns
response format (source citation) and domain register rather than broad factual recall.
Intended to be used as the generation slot of a RAG pipeline, not standalone.

## Links

- GitHub (docs & scripts): https://github.com/newcave/kwater-ailab-4b-lora
- Data inventory dashboard: https://github.com/newcave/rnddata_report
