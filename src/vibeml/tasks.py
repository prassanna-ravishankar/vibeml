"""Task generation functions for different training workflows."""

from typing import Dict, Optional, Callable, Any
import sky
from .exceptions import TaskGenerationError, ValidationError


def create_unsloth_task(
    model: str,
    dataset: str,
    gpu_type: str = "L40S",
    max_seq_length: int = 2048,
    lora_r: int = 16,
    max_steps: int = 60,
    learning_rate: float = 2e-4,
    output_dir: str = "./outputs",
    cloud_bucket: Optional[str] = None,
) -> sky.Task:
    """Generate SkyPilot Task for Unsloth fine-tuning on Nebius.

    Based on SkyPilot's Unsloth example but optimized for Nebius Cloud.

    Args:
        model: HuggingFace model ID (e.g., "mistralai/Mistral-7B-v0.1")
        dataset: HuggingFace dataset ID (e.g., "tatsu-lab/alpaca")
        gpu_type: Nebius GPU type (L40S, RTX4090, H100, A100)
        max_seq_length: Maximum sequence length for training
        lora_r: LoRA rank parameter
        max_steps: Maximum training steps
        learning_rate: Learning rate for training
        output_dir: Directory for saving outputs
        cloud_bucket: Optional cloud bucket for checkpoints

    Returns:
        sky.Task: Configured SkyPilot task for Nebius

    Raises:
        ValidationError: If inputs are invalid
        TaskGenerationError: If task generation fails
    """
    # Validate inputs
    if not model or "/" not in model:
        raise ValidationError(f"Invalid model format: {model}. Expected format: 'org/model'")

    if not dataset:
        raise ValidationError("Dataset cannot be empty")

    # Map GPU types to Nebius-specific configurations
    gpu_configs = {
        "L40S": {"accelerators": "L40S:1", "disk_size": 100},
        "RTX4090": {"accelerators": "RTX4090:1", "disk_size": 100},
        "H100": {"accelerators": "H100:1", "disk_size": 200},
        "A100": {"accelerators": "A100:1", "disk_size": 150},
    }

    if gpu_type not in gpu_configs:
        raise ValidationError(f"Unsupported GPU type: {gpu_type}. Choose from {list(gpu_configs.keys())}")

    config = gpu_configs[gpu_type]

    # Setup script - installs Unsloth with correct CUDA version
    setup_script = """
# Detect CUDA version and install appropriate packages
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[cu121_ampere_torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install xformers trl peft accelerate bitsandbytes datasets transformers
"""

    # Training script
    run_script = f"""
python -c "
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

print('Loading model: {model}')
model, tokenizer = FastLanguageModel.from_pretrained(
    '{model}',
    max_seq_length={max_seq_length},
    dtype=None,
    load_in_4bit=True,
)

print('Configuring LoRA')
model = FastLanguageModel.get_peft_model(
    model,
    r={lora_r},
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                   'gate_proj', 'up_proj', 'down_proj'],
    lora_alpha=16,
    lora_dropout=0,
    bias='none',
    use_gradient_checkpointing='unsloth',
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

print('Loading dataset: {dataset}')
dataset = load_dataset('{dataset}', split='train')

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field='text',
    max_seq_length={max_seq_length},
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps={max_steps},
        learning_rate={learning_rate},
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim='adamw_8bit',
        weight_decay=0.01,
        lr_scheduler_type='linear',
        seed=3407,
        output_dir='{output_dir}',
        save_strategy='steps',
        save_steps=50,
    ),
)

print('Starting training...')
trainer.train()

print('Saving model...')
model.save_pretrained('{output_dir}/final_model')
tokenizer.save_pretrained('{output_dir}/final_model')
print('Training complete!')
"
"""

    # Build file mounts if cloud bucket specified
    file_mounts = {}
    if cloud_bucket:
        file_mounts["/outputs"] = {"name": cloud_bucket, "mode": "MOUNT"}

    try:
        task = sky.Task(
            name=f"vibeml-unsloth-{model.replace('/', '-')}",
            setup=setup_script,
            run=run_script,
            resources=sky.Resources(
                cloud="nebius",
                region="eu-north1",
                accelerators=config["accelerators"],
                disk_size=config["disk_size"],
                use_spot=True,
            ),
        )

        if file_mounts:
            task.file_mounts = file_mounts

        return task

    except Exception as e:
        raise TaskGenerationError(f"Failed to create Unsloth task: {str(e)}")


def create_gpt_oss_lora_task(
    model_size: str = "20b",
    dataset: str = "tatsu-lab/alpaca",
    output_dir: str = "./outputs",
    cloud_bucket: Optional[str] = None,
) -> sky.Task:
    """Generate SkyPilot Task for GPT-OSS LoRA fine-tuning on Nebius.

    Based on SkyPilot's GPT-OSS example but for Nebius Cloud.

    Args:
        model_size: Model size - "20b" or "120b"
        dataset: HuggingFace dataset ID
        output_dir: Directory for saving outputs
        cloud_bucket: Optional cloud bucket for checkpoints

    Returns:
        sky.Task: Configured SkyPilot task

    Raises:
        ValidationError: If inputs are invalid
        TaskGenerationError: If task generation fails
    """
    # Model configurations for LoRA
    model_configs = {
        "20b": {
            "model": "EleutherAI/gpt-neox-20b",
            "accelerators": "H100:2",
            "disk_size": 200,
        },
        "120b": {
            "model": "EleutherAI/gpt-neox-120b",  # Placeholder
            "accelerators": "H100:8",
            "disk_size": 500,
        },
    }

    if model_size not in model_configs:
        raise ValidationError(f"Invalid model size: {model_size}. Choose from {list(model_configs.keys())}")

    config = model_configs[model_size]

    setup_script = """
pip install --upgrade pip
pip install torch transformers accelerate peft datasets bitsandbytes
pip install flash-attn --no-build-isolation
"""

    run_script = f"""
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer

print('Loading model: {config["model"]}')
model = AutoModelForCausalLM.from_pretrained(
    '{config["model"]}',
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained('{config["model"]}')
tokenizer.pad_token = tokenizer.eos_token

print('Configuring LoRA')
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=['query_key_value'],
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

print('Loading dataset: {dataset}')
dataset = load_dataset('{dataset}', split='train')

training_args = TrainingArguments(
    output_dir='{output_dir}',
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_strategy='steps',
    save_steps=100,
    evaluation_strategy='no',
    save_total_limit=3,
    load_best_model_at_end=False,
    ddp_find_unused_parameters=False,
    group_by_length=True,
    report_to='none',
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field='text',
    args=training_args,
    peft_config=peft_config,
)

print('Starting training...')
trainer.train()

print('Saving model...')
trainer.save_model('{output_dir}/final_model')
print('Training complete!')
"
"""

    file_mounts = {}
    if cloud_bucket:
        file_mounts["/outputs"] = {"name": cloud_bucket, "mode": "MOUNT"}

    try:
        task = sky.Task(
            name=f"vibeml-gpt-oss-lora-{model_size}",
            setup=setup_script,
            run=run_script,
            resources=sky.Resources(
                cloud="nebius",
                region="eu-north1",
                accelerators=config["accelerators"],
                disk_size=config["disk_size"],
                use_spot=True,
            ),
        )

        if file_mounts:
            task.file_mounts = file_mounts

        return task

    except Exception as e:
        raise TaskGenerationError(f"Failed to create GPT-OSS LoRA task: {str(e)}")


def create_gpt_oss_full_task(
    model_size: str = "20b",
    dataset: str = "tatsu-lab/alpaca",
    output_dir: str = "./outputs",
    cloud_bucket: Optional[str] = None,
) -> sky.Task:
    """Generate SkyPilot Task for GPT-OSS full fine-tuning on Nebius.

    Args:
        model_size: Model size - "20b" or "120b"
        dataset: HuggingFace dataset ID
        output_dir: Directory for saving outputs
        cloud_bucket: Optional cloud bucket for checkpoints

    Returns:
        sky.Task: Configured SkyPilot task

    Raises:
        ValidationError: If inputs are invalid
        TaskGenerationError: If task generation fails
    """
    # Model configurations for full fine-tuning
    model_configs = {
        "20b": {
            "model": "EleutherAI/gpt-neox-20b",
            "accelerators": "H100:8",
            "disk_size": 500,
            "num_nodes": 1,
        },
        "120b": {
            "model": "EleutherAI/gpt-neox-120b",  # Placeholder
            "accelerators": "H100:8",
            "disk_size": 1000,
            "num_nodes": 4,
        },
    }

    if model_size not in model_configs:
        raise ValidationError(f"Invalid model size: {model_size}. Choose from {list(model_configs.keys())}")

    config = model_configs[model_size]

    # For multi-node training, we'd need more complex setup
    # This is simplified for single-node
    setup_script = """
pip install --upgrade pip
pip install torch transformers accelerate datasets deepspeed
pip install flash-attn --no-build-isolation
"""

    run_script = f"""
echo "Full fine-tuning for {config["model"]} - requires distributed setup"
echo "This is a placeholder for full fine-tuning implementation"
echo "Would require DeepSpeed/FSDP configuration for production use"
"""

    try:
        resources = sky.Resources(
            cloud="nebius",
            region="eu-north1",
            accelerators=config["accelerators"],
            disk_size=config["disk_size"],
            use_spot=False,  # Full fine-tuning needs stable instances
        )

        task = sky.Task(
            name=f"vibeml-gpt-oss-full-{model_size}",
            setup=setup_script,
            run=run_script,
            resources=resources,
            num_nodes=config.get("num_nodes", 1),
        )

        return task

    except Exception as e:
        raise TaskGenerationError(f"Failed to create GPT-OSS full task: {str(e)}")


# Workflow registry
WORKFLOWS: Dict[str, Callable[..., sky.Task]] = {
    "unsloth": create_unsloth_task,
    "gpt-oss-lora": create_gpt_oss_lora_task,
    "gpt-oss-full": create_gpt_oss_full_task,
}


def get_workflow(workflow_type: str) -> Callable[..., sky.Task]:
    """Get workflow function by type.

    Args:
        workflow_type: Type of workflow

    Returns:
        Workflow function

    Raises:
        ValueError: If workflow type is unknown
    """
    if workflow_type not in WORKFLOWS:
        available = ", ".join(WORKFLOWS.keys())
        raise ValueError(f"Unknown workflow: {workflow_type}. Available: {available}")
    return WORKFLOWS[workflow_type]