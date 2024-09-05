# edge_llm_preliminary_submission
# Introduction
 
This repository has preliminary submission details for [**Edge-Device Large Language Model Competition**](https://github.com/TianjinYellow/EdgeDeviceLLMCompetition-Starting-Kit).
## Name of the team and team members 
### Byte Crunchers
- Chetan Singh Thakur
- Madhuvanthi Srivatsav R 
- Arun C R 
- Sriya R G  
- Dhruva Kashyap
 
## Track chosen: Compression challenge
### Method/Strategy:
#### Method 1:
Download model [Phi-2](https://huggingface.co/microsoft/phi-2) in fp16 format locally and wrap it for evaluation compatibility.
#### Method 2:
Structured pruning was performed on the provided base models [Phi-2](https://huggingface.co/microsoft/phi-2), [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) and [Qwen2-7B-Instruct]. (https://huggingface.co/Qwen/Qwen2-7B-Instruct).
[LLM-Pruner](https://arxiv.org/abs/2305.11627) adopts structural pruning that selectively removes non-critical coupled structures based on gradient information, maximally preserving the majority of the LLM's functionality.
 
# Installing
## Evaluation for CommonsenseQA, BIG-Bench Hard, GSM8K, LongBench, HumanEval, CHID, TruthfulQA Tasks
### Open Evaluation Task
The evaluation of CommonsenseQA, BIG-Bench Hard, GSM8K, HumanEval, CHID, and TruthfulQA is conducted using the OpenCompass tool.
#### Environment setup

```
  conda create --name opencompass python=3.10 
  conda activate opencompass
  conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
  pip install faiss-gpu

  # Install from source 
  git clone https://github.com/open-compass/opencompass opencompass
  cd opencompass
  git checkout 0.3.1
  pip install -e .

  # or with pip 
  pip install opencompass==0.3.1

  # Install human-eval
  pip install git+https://github.com/open-compass/human-eval.git
```
#### Data Preparation(Option-1)

If your environment cannot access the Internet, you can manually download the dataset.

```
# Download dataset to data/ folder
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
```
#### Data Preparation(Option-2)

The OpenCompass will automatically download the datasets either from its own server or from HuggingFace.


# Performance Evaluation Steps
### GPU Memory Usage and Throughput Measurement
```
# Replace the model/tokenizer loader code with your code. DO NOT CHANGE THE HYPER-PARAMETER SETTING.
python EvaluateThroughputAndMemory.py --model_name MODEL_NAME
```
-- batch_size needs to be set to 1 and max_length needs to be set to 2K.


 
# Accuracy Evaluation steps

### Evaluation Huggingface Models

Example Evaluation with 1 gpu
```
CUDA_VISIBLE_DEVICES=0 \
opencompass --datasets commonsenseqa_7shot_cot_gen_734a22 \ 
  FewCLUE_chid_gen \ 
  humaneval_gen \
  bbh_gen \
  gsm8k_gen \ 
  truthfulqa_gen \
  --hf-type chat \
  --hf-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --model-kwargs device_map='auto' trust_remote_code=True \
  --max-out-len 1024 \
  --debug \ 
  -r latest # You can add --dry-run to auto-download the datasets first before your evaluation
  # for Qwen2-7B-Instruct
  # --hf-path Qwen/Qwen2-7B-Instruct

```
Example Evaluation with multiple gpus
```
opencompass --datasets commonsenseqa_7shot_cot_gen_734a22 \
  FewCLUE_chid_gen \
  humaneval_gen \
  bbh_gen \
  gsm8k_gen \
  truthfulqa_gen \
  --hf-type chat \
  --hf-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --model-kwargs device_map='auto' trust_remote_code=True \
  --max-num-workers 8 \ ## Change this number based on number of gpus 
  --max-out-len 1024 \
  -r latest
  # for Qwen2-7B-Instruct
  # --hf-path Qwen/Qwen2-7B-Instruct
```

### Evaluate local models

Local model must be wrapped in the opencompass format. Refer to (https://opencompass.readthedocs.io/en/latest/advanced_guides/new_model.html).
Prepare the corresponding configuration file.
NOTE: The path of the saved model weights needs to specified in this configuration file. 

-- The wrapped model file (.py) needs to be placed under the folder: opencompass/opencompass/models.

-- The prepared configuration file needs be placed under the folder: /opencompass/configs. 
## Wrapper creation and usage
 
To wrap the local model so that it is compatible with opencompass, we followed the steps provided in [opencompass](https://opencompass.readthedocs.io/en/latest/advanced_guides/new_model.html)
 
1. Download phi-2 model in fp16 format
2. Add ``phi2custom.py`` to  ``<env's site_packages dir>/opencompass/models`` directory
3. Add a directory named `example_phi2` in ``<env's site_packages dir/opencompass/configs/models`` 
4. Add ``example_phi2.py`` to the above folder which contains configuration of the wrapped model
5. Insert ``from .phi2custom import Phi2Custom # noqa: F401`` to ``<env's site_packages dir>/opencompass/models/__init__.py`` for the created model to be identified
 
### Running the tasks
 
Tasks on the wrapped model were run using the following code
```
# remove -r latest if reusing previous examples is not intended
opencompass --datasets [name of the dataset] --hf-type chat \
--models example_phi2 --debug \
--model-kwargs device_map='auto' trust_remote_code=True \
--batch-size 1 -r latest --max-out-len 1024 --max-num-workers 2
```
 
 
# Submission status as of 5/9/24
1. Phi-2 model (fp16) has been evaluated locally and submitted.
2. The above model is pending MLC compilation and will be submitted in a future submission 
3. Other two model submissions are a work in progress

## Preliminary Results for phi-2

### Evaluation Results Summary (5/9/24)

| Metric             | Result |
|--------------------|--------|
| CommonsenseQA      | 19.57  |
| BIG-Bench-Hard     | 0      |
| GSM8K              | 0      |
| HumanEval          | 31.1   |
| CHID               | 12.29  |
| TruthfulQA         | 0.18   |
| Throughput         | 0      |
| Memory-Usage       | 0      |

#### Notes

- hftype chat was used while generating results with batch size 1 and max output length 1024.

#### System Configuration

- **CPU Configuration**: 2P 96C Genoa AMD CPU
- **System RAM**: 1.5 TB
- **System Storage**: 48 TB
- **GPU Configuration**: 4 x AMD MI210 (64 GB)
- **Operating System**: Ubuntu 22.04 LTS
- **ROCm Version**: 5.7


