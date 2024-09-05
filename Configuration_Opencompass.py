from opencompass.models import Phi2Custom

models = [
    dict(
        type=Phi2Custom,
        # Parameters for `HuggingFaceCausalLM` initialization.
        path='/rhome/sriyar/Atul/opencompass/phi-2-fp16',
        tokenizer_path='microsoft/phi-2',
        #max_seq_len=2048,
        #batch_padding=False,
        #peft_path='/rhome/sriyar/Sriya/LLM-Pruner/tune_log/tuned_llama',
        # Common parameters shared by various models, not specific to `HuggingFaceCausalLM` initialization.
        abbr='example_phi2',            # Model abbreviation used for result display.
        max_out_len=256,            # Maximum number of generated tokens.
        batch_size=1,              # The size of a batch during inference.
        run_cfg=dict(num_gpus=1),   # Run configuration to specify resource requirements.
    )
]