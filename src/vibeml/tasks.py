"""Task generation functions for different training workflows."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import sky

from .exceptions import TaskGenerationError, ValidationError

NEBIUS_INFRA = "nebius/eu-north1"

# Where training writes checkpoints and final models, and where a cloud bucket
# is mounted. One path for both, so outputs always land in the bucket.
OUTPUTS_DIR = "/outputs"


@dataclass(frozen=True)
class Compute:
    """GPU resources a workflow configuration runs on."""

    accelerators: str
    disk_size: int
    num_nodes: int = 1


@dataclass(frozen=True)
class ModelConfig:
    """A model and the compute it trains on."""

    model: str
    compute: Compute


def _nebius_task(
    name: str,
    install: str,
    run: str,
    compute: Compute,
    *,
    use_spot: bool,
    output_dir: str,
    cloud_bucket: str | None,
    envs: dict[str, str],
) -> sky.Task:
    """Build a task on Nebius whose outputs go to ``output_dir``.

    User-supplied values reach the scripts only as environment variables
    (``envs`` plus ``VIBEML_OUTPUT_DIR``), which SkyPilot exports with shell
    quoting, so they are never interpreted as code.

    Dependency installs run at the start of ``run`` rather than in setup:
    SkyPilot arms autostop (vibeml's cost cap) only after setup finishes, so
    a slow or hanging install must happen under the armed timer.

    With ``cloud_bucket``, the bucket is mounted at ``output_dir``, which is
    where the run script writes. Without one, setup creates the directory if
    it does not exist yet.

    Args:
        name: Task name.
        install: Dependency installation commands.
        run: Run script; writes outputs under ``$VIBEML_OUTPUT_DIR``.
        compute: GPU resources.
        use_spot: Whether to use spot instances.
        output_dir: Absolute path for outputs, and the bucket mount point.
        cloud_bucket: Bucket to mount at ``output_dir``, if any.
        envs: Other user-supplied values the scripts read.

    Returns:
        The configured task.
    """
    setup = None
    if not cloud_bucket:
        setup = (
            'if [ ! -e "$VIBEML_OUTPUT_DIR" ]; then\n'
            '  sudo mkdir -p -- "$VIBEML_OUTPUT_DIR"\n'
            '  sudo chown "$(id -u):$(id -g)" -- "$VIBEML_OUTPUT_DIR"\n'
            "fi\n"
        )
    task = sky.Task(
        name=name,
        setup=setup,
        run=install + run,
        envs={**envs, "VIBEML_OUTPUT_DIR": output_dir},
        num_nodes=compute.num_nodes,
        resources=sky.Resources(
            infra=NEBIUS_INFRA,
            accelerators=compute.accelerators,
            disk_size=compute.disk_size,
            use_spot=use_spot,
        ),
    )
    if cloud_bucket:
        task.set_storage_mounts(
            {output_dir: sky.Storage(name=cloud_bucket, mode=sky.StorageMode.MOUNT)}
        )
    return task


def _check_output_dir(output_dir: str) -> None:
    """Require an absolute output directory, since it is also a mount point.

    Args:
        output_dir: Directory the run script writes outputs to.

    Raises:
        ValidationError: If the path is relative.
    """
    if not output_dir.startswith("/"):
        raise ValidationError(
            f"output_dir must be an absolute path (it is also the bucket mount point), got {output_dir!r}"
        )


def create_unsloth_task(
    model: str,
    dataset: str,
    gpu_type: str = "L40S",
    max_seq_length: int = 2048,
    lora_r: int = 16,
    max_steps: int = 60,
    learning_rate: float = 2e-4,
    output_dir: str = OUTPUTS_DIR,
    cloud_bucket: str | None = None,
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
        output_dir: Absolute directory for outputs; the bucket mount point
        cloud_bucket: Optional cloud bucket for checkpoints

    Returns:
        sky.Task: Configured SkyPilot task for Nebius

    Raises:
        ValidationError: If inputs are invalid
        TaskGenerationError: If task generation fails
    """
    # Validate inputs
    _check_output_dir(output_dir)
    if not model or "/" not in model:
        raise ValidationError(
            f"Invalid model format: {model}. Expected format: 'org/model'"
        )

    if not dataset:
        raise ValidationError("Dataset cannot be empty")

    # Map GPU types to Nebius-specific configurations
    gpu_configs = {
        "L40S": Compute("L40S:1", disk_size=100),
        "RTX4090": Compute("RTX4090:1", disk_size=100),
        "H100": Compute("H100:1", disk_size=200),
        "A100": Compute("A100:1", disk_size=150),
    }

    if gpu_type not in gpu_configs:
        raise ValidationError(
            f"Unsupported GPU type: {gpu_type}. Choose from {list(gpu_configs.keys())}"
        )

    compute = gpu_configs[gpu_type]

    # Setup script - installs Unsloth with correct CUDA version
    install_script = """
# Detect CUDA version and install appropriate packages
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[cu121_ampere_torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install xformers trl peft accelerate bitsandbytes datasets transformers
"""

    # Training script
    run_script = f"""
python -c "
import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

print('Loading model:', os.environ['VIBEML_MODEL'])
model, tokenizer = FastLanguageModel.from_pretrained(
    os.environ['VIBEML_MODEL'],
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

print('Loading dataset:', os.environ['VIBEML_DATASET'])
dataset = load_dataset(os.environ['VIBEML_DATASET'], split='train')

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
        output_dir=os.environ['VIBEML_OUTPUT_DIR'],
        save_strategy='steps',
        save_steps=50,
    ),
)

print('Starting training...')
trainer.train()

print('Saving model...')
model.save_pretrained(os.path.join(os.environ['VIBEML_OUTPUT_DIR'], 'final_model'))
tokenizer.save_pretrained(os.path.join(os.environ['VIBEML_OUTPUT_DIR'], 'final_model'))
print('Training complete!')
"
"""

    try:
        return _nebius_task(
            f"vibeml-unsloth-{model.replace('/', '-')}",
            install_script,
            run_script,
            compute,
            use_spot=True,
            output_dir=output_dir,
            cloud_bucket=cloud_bucket,
            envs={"VIBEML_MODEL": model, "VIBEML_DATASET": dataset},
        )

    except Exception as e:
        raise TaskGenerationError(f"Failed to create Unsloth task: {e!s}") from e


def create_gpt_oss_lora_task(
    model_size: str = "20b",
    dataset: str = "tatsu-lab/alpaca",
    output_dir: str = OUTPUTS_DIR,
    cloud_bucket: str | None = None,
) -> sky.Task:
    """Generate SkyPilot Task for GPT-OSS LoRA fine-tuning on Nebius.

    Based on SkyPilot's GPT-OSS example but for Nebius Cloud.

    Args:
        model_size: Model size - "20b" or "120b"
        dataset: HuggingFace dataset ID
        output_dir: Absolute directory for outputs; the bucket mount point
        cloud_bucket: Optional cloud bucket for checkpoints

    Returns:
        sky.Task: Configured SkyPilot task

    Raises:
        ValidationError: If inputs are invalid
        TaskGenerationError: If task generation fails
    """
    # Model configurations for LoRA
    model_configs = {
        "20b": ModelConfig("EleutherAI/gpt-neox-20b", Compute("H100:2", disk_size=200)),
        "120b": ModelConfig(
            "EleutherAI/gpt-neox-120b",  # Placeholder
            Compute("H100:8", disk_size=500),
        ),
    }

    _check_output_dir(output_dir)
    if model_size not in model_configs:
        raise ValidationError(
            f"Invalid model size: {model_size}. Choose from {list(model_configs.keys())}"
        )

    config = model_configs[model_size]

    install_script = """
pip install --upgrade pip
pip install torch transformers accelerate peft datasets bitsandbytes
pip install flash-attn --no-build-isolation
"""

    run_script = f"""
python -c "
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer

print('Loading model: {config.model}')
model = AutoModelForCausalLM.from_pretrained(
    '{config.model}',
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained('{config.model}')
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

print('Loading dataset:', os.environ['VIBEML_DATASET'])
dataset = load_dataset(os.environ['VIBEML_DATASET'], split='train')

training_args = TrainingArguments(
    output_dir=os.environ['VIBEML_OUTPUT_DIR'],
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
trainer.save_model(os.path.join(os.environ['VIBEML_OUTPUT_DIR'], 'final_model'))
print('Training complete!')
"
"""

    try:
        return _nebius_task(
            f"vibeml-gpt-oss-lora-{model_size}",
            install_script,
            run_script,
            config.compute,
            use_spot=True,
            output_dir=output_dir,
            cloud_bucket=cloud_bucket,
            envs={"VIBEML_DATASET": dataset},
        )

    except Exception as e:
        raise TaskGenerationError(f"Failed to create GPT-OSS LoRA task: {e!s}") from e


def create_gpt_oss_full_task(
    model_size: str = "20b",
    dataset: str = "tatsu-lab/alpaca",
    output_dir: str = OUTPUTS_DIR,
    cloud_bucket: str | None = None,
) -> sky.Task:
    """Generate SkyPilot Task for GPT-OSS full fine-tuning on Nebius.

    Args:
        model_size: Model size - "20b" or "120b"
        dataset: HuggingFace dataset ID
        output_dir: Absolute directory for outputs; the bucket mount point
        cloud_bucket: Optional cloud bucket for checkpoints

    Returns:
        sky.Task: Configured SkyPilot task

    Raises:
        ValidationError: If inputs are invalid
        TaskGenerationError: If task generation fails
    """
    # Model configurations for full fine-tuning
    model_configs = {
        "20b": ModelConfig("EleutherAI/gpt-neox-20b", Compute("H100:8", disk_size=500)),
        "120b": ModelConfig(
            "EleutherAI/gpt-neox-120b",  # Placeholder
            Compute("H100:8", disk_size=1000, num_nodes=4),
        ),
    }

    _check_output_dir(output_dir)
    if model_size not in model_configs:
        raise ValidationError(
            f"Invalid model size: {model_size}. Choose from {list(model_configs.keys())}"
        )

    config = model_configs[model_size]

    # For multi-node training, we'd need more complex setup
    # This is simplified for single-node
    install_script = """
pip install --upgrade pip
pip install torch transformers accelerate datasets deepspeed
pip install flash-attn --no-build-isolation
"""

    run_script = f"""
echo "Full fine-tuning for {config.model} - requires distributed setup"
echo "This is a placeholder for full fine-tuning implementation"
echo "Would require DeepSpeed/FSDP configuration for production use"
"""

    try:
        return _nebius_task(
            f"vibeml-gpt-oss-full-{model_size}",
            install_script,
            run_script,
            config.compute,
            use_spot=False,  # Full fine-tuning needs stable instances
            output_dir=output_dir,
            cloud_bucket=cloud_bucket,
            envs={"VIBEML_DATASET": dataset},
        )

    except Exception as e:
        raise TaskGenerationError(f"Failed to create GPT-OSS full task: {e!s}") from e


# Workflow registry
WORKFLOWS: dict[str, Callable[..., sky.Task]] = {
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


def build_task(
    workflow: str,
    model: str,
    dataset: str,
    gpu_type: str | None = None,
    **hyperparameters: Any,
) -> sky.Task:
    """Build the SkyPilot task for a workflow from tool-level parameters.

    Args:
        workflow: Workflow name, a key of ``WORKFLOWS``.
        model: HuggingFace model ID, or the model size ("20b", "120b") for
            GPT-OSS workflows.
        dataset: HuggingFace dataset ID.
        gpu_type: GPU type for Unsloth; GPT-OSS sizes fix their own GPUs.
        **hyperparameters: Optional workflow arguments; None values are ignored.

    Returns:
        The configured task.

    Raises:
        ValidationError: If the workflow is unknown or its inputs are invalid.
    """
    try:
        create = get_workflow(workflow)
    except ValueError as e:
        raise ValidationError(str(e)) from e

    given = {k: v for k, v in hyperparameters.items() if v is not None}
    if workflow == "unsloth":
        return create(
            model=model, dataset=dataset, gpu_type=gpu_type or "L40S", **given
        )
    shared = {k: given[k] for k in ("output_dir", "cloud_bucket") if k in given}
    return create(model_size=model, dataset=dataset, **shared)
