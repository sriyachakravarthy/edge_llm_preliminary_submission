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
### Strategy:
#### For preliminary Submission:
   Download model [Phi-2](https://huggingface.co/microsoft/phi-2) in fp16 format locally and wrap it for evaluation compatibility.
#### Strategy for final submission:
   Structured pruning will be performed on the provided base models [Phi-2](https://huggingface.co/microsoft/phi-2), [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) and   [Qwen2-7B-Instruct]. (https://huggingface.co/Qwen/Qwen2-7B-Instruct).
   We plan to utilize [LLM-Pruner](https://arxiv.org/abs/2305.11627) which adopts structural pruning that selectively removes non-critical coupled structures based on gradient information, maximally preserving the majority of the LLM's functionality.
 







### Evaluate local models

Local model must be wrapped in the opencompass format. Refer to (https://opencompass.readthedocs.io/en/latest/advanced_guides/new_model.html).
Prepare the corresponding configuration file.
NOTE: The path of the saved model weights needs to specified in this configuration file. 

-- The wrapped model file (.py) needs to be placed under the folder: opencompass/opencompass/models.

-- The prepared configuration file needs be placed under the folder: /opencompass/configs. 
## Wrapper creation and usage
 
To wrap the local model so that it is compatible with opencompass, we followed the steps provided in [opencompass](https://opencompass.readthedocs.io/en/latest/advanced_guides/new_model.html)
 
1. Download phi-2 model in fp16 format using code provided in source_code folder
2. Download Opencompass_model.py as phi2custom.py ``phi2custom.py`` to  ``<env's site_packages dir>/opencompass/models`` directory
3. Add a directory named `example_phi2` in ``<env's site_packages dir/opencompass/configs/models`` 
4. Download Configuration_opencompass.py as ``example_phi2.py`` to the above folder which contains configuration of the wrapped model
5. Make sure that the 'save_dir' in step1 is set as path in example_phi2.py
6. Insert ``from .phi2custom import Phi2Custom # noqa: F401`` to ``<env's site_packages dir>/opencompass/models/__init__.py`` for the created model to be identified
 
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
| Throughput         | 34.66      |
| Memory-Usage       | 6.534 GB      |

#### Notes

- hftype chat was used while generating results with batch size 1 and max output length 1024.

#### System Configuration

- **CPU Configuration**: 2P 96C Genoa AMD CPU
- **System RAM**: 1.5 TB
- **System Storage**: 48 TB
- **GPU Configuration**: 4 x AMD MI210 (64 GB)
- **Operating System**: Ubuntu 22.04 LTS
- **ROCm Version**: 5.7

-The Throughput and memory evaluation numbers from obtained from Nvidia A100 GPU with the following specs:

- CPU Configuration: AMD EPYC 7742 64-Core Processor    
- System RAM: 512 GB
- GPU Configuration: 4 x Nvidia A100 (40 GB)
- Operating System: Ubuntu 22.04.4 LTS
- CUDA Version: 12.4


