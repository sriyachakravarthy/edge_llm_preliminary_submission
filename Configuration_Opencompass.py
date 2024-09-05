from opencompass.models import Phi2Custom

models = [
    dict(
        type=Phi2Custom,
        # Parameters for `HuggingFaceCausalLM` initialization.
        path='/rhome/sriyar/Atul/opencompass/phi-2-fp16', #the path to the directory in which the model is loaded to
        tokenizer_path='microsoft/phi-2',
        abbr='example_phi2',            # Model abbreviation used for result display.
        max_out_len=256,            # Maximum number of generated tokens.
        batch_size=1,              # The size of a batch during inference.
        run_cfg=dict(num_gpus=1),   # Run configuration to specify resource requirements.
    )
]
