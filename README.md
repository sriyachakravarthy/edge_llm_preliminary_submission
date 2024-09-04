# edge_llm_preliminary_submission
# Introduction
 
This repository has preliminary submission details for **Edge-Device Large Language Model Competition**.
## Name of the team and team members 
### Byte Crunchers
- Chetan Singh Thakur
- Madhu 
- Arun 
- Sriya 
- Dhruva
 
## Track chosen: Compression challenge
### Method/Strategy:
Structured pruning was performed on the provided base models [Phi-2](https://huggingface.co/microsoft/phi-2), [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) and [Qwen2-7B-Instruct]. (https://huggingface.co/Qwen/Qwen2-7B-Instruct).
[LLM-Pruner](https://arxiv.org/abs/2305.11627) adopts structural pruning that selectively removes non-critical coupled structures based on gradient information, maximally preserving the majority of the LLM's functionality.
 
# Installing
 
 
# Performance Evaluation Steps
 
# Accuracy Evaluation steps
 
## Wrapper creation and usage
 
To wrap the pruned model so that it is compatible with opencompass, we followed the steps provided in [opencompass](https://opencompass.readthedocs.io/en/latest/advanced_guides/new_model.html)
 
1. Add ``mymodel.py`` to  ``<env's site_packages dir>/opencompass/models`` directory
2. Add a directory named `example` in ``<env's site_packages dir/opencompass/configs/models`` 
3. Add ``example.py`` to the above folder which contains configuration of the wrapped model
4. Insert ``from .mymodel import MyModel # noqa: F401`` to ``<env's site_packages dir>/opencompass/models/__init__.py`` for the created model to be identified
 
## Running the tasks
 
Tasks on the wrapped model were run using the following code
```
opencompass --datasets [name of the dataset] --hf-type chat \
--models example --debug \
--model-kwargs device_map='auto' trust_remote_code=True \
--batch-size 1 -r latest --max-out-len 1024 --max-num-workers 2
```
 
 
# Limitations for this release (MLC models are not compiled, pruned models are not fine-tuned, only 1 / 2 models submitted (depending on team))
