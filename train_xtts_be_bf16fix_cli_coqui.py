from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Iterable

import torch
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsAudioConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.utils.manage import ModelManager

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset-1"
TRAIN_META = DATASET_DIR / "metadata_train.csv"
EVAL_META = DATASET_DIR / "metadata_eval.csv"

# Coqui formatter metadata with header:
#   audio_file|text|speaker_name
# Example:
#   audio_file|text|speaker_name
#   wavs/utt_0001.wav|Прывітанне, свет!|speaker_1
RUN_NAME = "GPT_XTTS_BE_FT"
PROJECT_NAME = "XTTS_trainer"
DASHBOARD_LOGGER = "tensorboard"
LOGGER_URI = None

DEFAULT_OUT_PATH = PROJECT_ROOT / "run" / "training"
CHECKPOINTS_OUT_PATH = PROJECT_ROOT / "checkpoints" / "XTTS_base"
CHECKPOINTS_OUT_PATH.mkdir(parents=True, exist_ok=True)
DEFAULT_OUT_PATH.mkdir(parents=True, exist_ok=True)

LANGUAGE = "be"
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True
TEST_SENTENCE = "Гэта тэставае сказанне для праверкі беларускага XTTS пасля fine-tuning."
HEADER_TOKENS = {"audio_file", "wav_file", "file", "path", "id", "filename"}
COQUI_REQUIRED_COLUMNS = ("audio_file", "text")
COQUI_OPTIONAL_COLUMNS = ("speaker_name",)

# Runtime-configurable globals initialized from CLI/env in main().
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 20
NUM_EPOCHS = 1
LEARNING_RATE = 9e-6
SAVE_STEP = 540
START_WITH_EVAL = False
USE_AMP = True
USE_TORCH_COMPILE = False
TORCH_COMPILE_MODE = "default"
TORCH_COMPILE_BACKEND = "inductor"
ENABLE_TF32_MATMUL = True
ENABLE_TF32_CUDNN = True
ENABLE_CUDNN_BENCHMARK = True
ENABLE_SDP_KERNELS = True
NUM_LOADER_WORKERS = 2
PREFETCH_FACTOR = 2
PIN_MEMORY = True
PERSISTENT_WORKERS = True
AMP_PRECISION = "fp16"
OUT_PATH = DEFAULT_OUT_PATH
USE_AUGMENTATION = False
MONITOR_GRADIENTS = False


def normalize_amp_precision_name(value: str) -> str:
    value = (value or "").strip().lower()
    mapping = {
        "bfloat16": "bf16",
        "bf16": "bf16",
        "float16": "fp16",
        "fp16": "fp16",
        "float32": "float32",
        "fp32": "float32",
        "32": "float32",
    }
    return mapping.get(value, value)



def preferred_amp_precision(raw_override: str | None = None, force_bf16: bool = False) -> str:
    if force_bf16:
        return "bf16"

    env_override = raw_override or os.environ.get("XTTS_AMP_PRECISION")
    if env_override:
        return normalize_amp_precision_name(env_override)

    if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported"):
        try:
            if torch.cuda.is_bf16_supported():
                return "bf16"
        except Exception:
            pass

    return "fp16"



def set_attr_if_present(obj, attr_name: str, value) -> bool:
    if hasattr(obj, attr_name):
        setattr(obj, attr_name, value)
        return True
    return False



def resolve_output_path(value: str | None) -> Path:
    if not value:
        return DEFAULT_OUT_PATH
    p = Path(value)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XTTS Belarusian training launcher with Coqui formatter and CLI speed/runtime knobs")
    parser.add_argument("--output_path", type=str, default=os.environ.get("XTTS_OUTPUT_PATH", str(DEFAULT_OUT_PATH)))
    parser.add_argument("--batch_size", type=int, default=int(os.environ.get("XTTS_BATCH_SIZE", "2")))
    parser.add_argument("--grad_acumm", "--grad_accum", dest="grad_accum", type=int, default=int(os.environ.get("XTTS_GRAD_ACCUM_STEPS", "20")))
    parser.add_argument("--save_step", type=int, default=int(os.environ.get("XTTS_SAVE_STEP", "540")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("XTTS_LR", "9e-6")))
    parser.add_argument("--num_epochs", type=int, default=int(os.environ.get("XTTS_NUM_EPOCHS", "1")))

    parser.add_argument("--use_augmentation", action="store_true")
    parser.add_argument("--monitor_gradients", action="store_true")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--use_bfloat16", action="store_true")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--no_mixed_precision", action="store_true")
    parser.add_argument("--start_with_eval", action="store_true")

    parser.add_argument("--tf32_matmul", action="store_true")
    parser.add_argument("--tf32_cudnn", action="store_true")
    parser.add_argument("--disable_tf32", action="store_true")
    parser.add_argument("--cudnn_benchmark", action="store_true")
    parser.add_argument("--disable_cudnn_benchmark", action="store_true")
    parser.add_argument("--disable_sdp_kernels", action="store_true")

    parser.add_argument("--num_loader_workers", type=int, default=int(os.environ.get("XTTS_NUM_LOADER_WORKERS", "2")))
    parser.add_argument("--prefetch_factor", type=int, default=int(os.environ.get("XTTS_PREFETCH_FACTOR", "2")))
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--no_persistent_workers", action="store_true")

    parser.add_argument("--amp_precision", type=str, default=os.environ.get("XTTS_AMP_PRECISION"))
    parser.add_argument("--torch_compile_mode", type=str, default=os.environ.get("XTTS_TORCH_COMPILE_MODE", "default"))
    parser.add_argument("--torch_compile_backend", type=str, default=os.environ.get("XTTS_TORCH_COMPILE_BACKEND", "inductor"))
    return parser



def apply_cli_args(args: argparse.Namespace) -> None:
    global BATCH_SIZE, GRAD_ACCUM_STEPS, NUM_EPOCHS, LEARNING_RATE, SAVE_STEP, START_WITH_EVAL
    global USE_AMP, USE_TORCH_COMPILE, TORCH_COMPILE_MODE, TORCH_COMPILE_BACKEND
    global ENABLE_TF32_MATMUL, ENABLE_TF32_CUDNN, ENABLE_CUDNN_BENCHMARK, ENABLE_SDP_KERNELS
    global NUM_LOADER_WORKERS, PREFETCH_FACTOR, PIN_MEMORY, PERSISTENT_WORKERS
    global AMP_PRECISION, OUT_PATH, USE_AUGMENTATION, MONITOR_GRADIENTS

    BATCH_SIZE = args.batch_size
    GRAD_ACCUM_STEPS = args.grad_accum
    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.lr
    SAVE_STEP = args.save_step
    START_WITH_EVAL = args.start_with_eval or os.environ.get("XTTS_START_WITH_EVAL", "0") == "1"

    USE_AUGMENTATION = args.use_augmentation
    MONITOR_GRADIENTS = args.monitor_gradients

    USE_TORCH_COMPILE = args.compile_model or os.environ.get("XTTS_USE_TORCH_COMPILE", "0") == "1"
    TORCH_COMPILE_MODE = args.torch_compile_mode
    TORCH_COMPILE_BACKEND = args.torch_compile_backend

    if args.disable_tf32:
        ENABLE_TF32_MATMUL = False
        ENABLE_TF32_CUDNN = False
    else:
        env_tf32 = os.environ.get("XTTS_ENABLE_TF32", "1") == "1"
        ENABLE_TF32_MATMUL = args.tf32_matmul or env_tf32
        ENABLE_TF32_CUDNN = args.tf32_cudnn or env_tf32

    if args.disable_cudnn_benchmark:
        ENABLE_CUDNN_BENCHMARK = False
    else:
        ENABLE_CUDNN_BENCHMARK = args.cudnn_benchmark or os.environ.get("XTTS_CUDNN_BENCHMARK", "1") == "1"

    ENABLE_SDP_KERNELS = not args.disable_sdp_kernels and os.environ.get("XTTS_ENABLE_SDP_KERNELS", "1") == "1"

    NUM_LOADER_WORKERS = args.num_loader_workers
    PREFETCH_FACTOR = args.prefetch_factor

    if args.no_pin_memory:
        PIN_MEMORY = False
    elif args.pin_memory:
        PIN_MEMORY = True
    else:
        PIN_MEMORY = os.environ.get("XTTS_PIN_MEMORY", "1") == "1"

    if args.no_persistent_workers:
        PERSISTENT_WORKERS = False
    elif args.persistent_workers:
        PERSISTENT_WORKERS = True
    else:
        PERSISTENT_WORKERS = os.environ.get("XTTS_PERSISTENT_WORKERS", "1") == "1"

    USE_AMP = not args.no_mixed_precision
    if args.use_bfloat16:
        AMP_PRECISION = "bf16"
        USE_AMP = True
    elif args.use_fp16:
        AMP_PRECISION = "fp16"
        USE_AMP = True
    else:
        AMP_PRECISION = preferred_amp_precision(raw_override=args.amp_precision)

    OUT_PATH = resolve_output_path(args.output_path)
    OUT_PATH.mkdir(parents=True, exist_ok=True)



def print_launch_config(args: argparse.Namespace) -> None:
    print("Launch config:")
    print(f"  output_path = {OUT_PATH}")
    print(f"  batch_size = {BATCH_SIZE}")
    print(f"  grad_accum_steps = {GRAD_ACCUM_STEPS}")
    print(f"  save_step = {SAVE_STEP}")
    print(f"  lr = {LEARNING_RATE}")
    print(f"  num_epochs = {NUM_EPOCHS}")
    print(f"  use_augmentation = {USE_AUGMENTATION}")
    print(f"  monitor_gradients = {MONITOR_GRADIENTS}")
    print(f"  compile_model = {USE_TORCH_COMPILE}")
    print(f"  tf32_matmul = {ENABLE_TF32_MATMUL}")
    print(f"  tf32_cudnn = {ENABLE_TF32_CUDNN}")
    print(f"  use_amp = {USE_AMP}")
    print(f"  amp_precision = {AMP_PRECISION}")
    print(f"  num_loader_workers = {NUM_LOADER_WORKERS}")



def enable_tf32_if_supported() -> bool:
    if not (ENABLE_TF32_MATMUL or ENABLE_TF32_CUDNN):
        print("TF32 disabled by CLI / env settings")
        return False

    if not torch.cuda.is_available():
        print("TF32 not enabled: CUDA is not available.")
        return False

    supported = False

    if hasattr(torch.cuda, "is_tf32_supported"):
        try:
            supported = bool(torch.cuda.is_tf32_supported())
        except Exception:
            supported = False

    if not supported:
        try:
            major, _minor = torch.cuda.get_device_capability()
            supported = major >= 8
        except Exception:
            supported = False

    if not supported:
        print("TF32 not enabled: current GPU / PyTorch build does not support TF32.")
        return False

    try:
        if ENABLE_TF32_MATMUL and hasattr(torch.backends, "fp32_precision"):
            torch.backends.fp32_precision = "tf32"
        if ENABLE_TF32_MATMUL and hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "tf32"
        if ENABLE_TF32_CUDNN and hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "fp32_precision"):
            torch.backends.cudnn.fp32_precision = "tf32"
    except Exception as exc:
        warnings.warn(f"New TF32 precision API could not be applied cleanly: {exc}")

    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(ENABLE_TF32_MATMUL)
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = bool(ENABLE_TF32_CUDNN)
    except Exception:
        pass

    if ENABLE_TF32_MATMUL and hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception as exc:
            warnings.warn(f"torch.set_float32_matmul_precision failed: {exc}")

    print("TF32 configured.")
    try:
        print("TF32 supported:", torch.cuda.is_tf32_supported())
    except Exception:
        print("TF32 supported:", supported)
    try:
        print("cuda matmul allow_tf32:", torch.backends.cuda.matmul.allow_tf32)
    except Exception:
        pass
    try:
        print("cudnn allow_tf32:", torch.backends.cudnn.allow_tf32)
    except Exception:
        pass
    if hasattr(torch, "get_float32_matmul_precision"):
        try:
            print("float32 matmul precision:", torch.get_float32_matmul_precision())
        except Exception:
            pass
    return True



def enable_runtime_speedups() -> None:
    enable_tf32_if_supported()

    if ENABLE_CUDNN_BENCHMARK:
        try:
            torch.backends.cudnn.benchmark = True
            print("cuDNN benchmark enabled.")
        except Exception as exc:
            warnings.warn(f"Could not enable cuDNN benchmark: {exc}")
    else:
        try:
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

    if ENABLE_SDP_KERNELS and hasattr(torch.backends, "cuda"):
        for fn_name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp"):
            fn = getattr(torch.backends.cuda, fn_name, None)
            if callable(fn):
                try:
                    fn(True)
                    print(f"{fn_name}(True)")
                except Exception as exc:
                    warnings.warn(f"Could not call {fn_name}(True): {exc}")

    print(f"AMP requested: {USE_AMP}")
    print(f"AMP precision (trainer): {AMP_PRECISION}")
    print(f"Torch compile requested: {USE_TORCH_COMPILE}")



def first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path and path.exists():
            return path
    return None



def read_metadata_lines(meta_path: Path) -> list[str]:
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    rows = [line.strip() for line in meta_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"No rows found in {meta_path}")
    return rows



def parse_metadata_header(meta_path: Path) -> list[str]:
    rows = read_metadata_lines(meta_path)
    header = [c.strip() for c in rows[0].split("|")]
    if not header or header[0].lower() not in HEADER_TOKENS:
        raise RuntimeError(
            f"{meta_path.name} must use Coqui formatter metadata with a header row like: "
            "audio_file|text|speaker_name"
        )
    return header



def iter_metadata_rows(meta_path: Path) -> Iterable[list[str]]:
    rows = read_metadata_lines(meta_path)

    for i, row in enumerate(rows):
        cols = [c.strip() for c in row.split("|")]
        if not cols:
            continue
        if i == 0 and cols[0].lower() in HEADER_TOKENS:
            continue
        yield cols



def resolve_dataset_wav(wav_value: str, dataset_dir: Path) -> Path:
    p = Path(wav_value)
    name = p.name
    stem = p.stem

    candidates = [
        dataset_dir / wav_value,
        dataset_dir / name,
        dataset_dir / "wavs" / name,
        dataset_dir / "wavs" / f"{stem}.wav",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"WAV file referenced in metadata was not found: {wav_value}\n"
        f"Tried: {[str(x) for x in candidates]}"
    )



def validate_coqui_metadata(meta_path: Path, dataset_dir: Path) -> None:
    header = parse_metadata_header(meta_path)
    header_to_idx = {name.strip(): idx for idx, name in enumerate(header)}

    missing_required = [name for name in COQUI_REQUIRED_COLUMNS if name not in header_to_idx]
    if missing_required:
        raise RuntimeError(
            f"{meta_path.name} is missing required Coqui metadata columns: {missing_required}. "
            "Expected header like: audio_file|text|speaker_name"
        )

    audio_idx = header_to_idx["audio_file"]
    text_idx = header_to_idx["text"]
    speaker_idx = header_to_idx.get("speaker_name")

    checked = 0
    speaker_values: set[str] = set()

    for row_num, cols in enumerate(iter_metadata_rows(meta_path), start=2):
        max_required_idx = max(audio_idx, text_idx, speaker_idx or 0)
        if len(cols) <= max_required_idx:
            raise RuntimeError(
                f"{meta_path.name}:{row_num} has too few columns for header {header}. "
                f"Row values: {cols}"
            )

        wav_value = cols[audio_idx].strip()
        text_value = cols[text_idx].strip()
        if not wav_value:
            raise RuntimeError(f"{meta_path.name}:{row_num} has empty audio_file")
        if not text_value:
            raise RuntimeError(f"{meta_path.name}:{row_num} has empty text")

        resolve_dataset_wav(wav_value, dataset_dir)

        if speaker_idx is not None and speaker_idx < len(cols):
            speaker_value = cols[speaker_idx].strip()
            if speaker_value:
                speaker_values.add(speaker_value)

        checked += 1

    if checked == 0:
        raise RuntimeError(f"No usable rows found in {meta_path.name}")

    speaker_info = f", unique speakers = {len(speaker_values)}" if speaker_idx is not None else ", speaker_name column missing"
    print(f"Validated Coqui metadata: {meta_path.name} ({checked} rows{speaker_info})")



def parse_reference_wav(meta_path: Path, dataset_dir: Path) -> str:
    header = parse_metadata_header(meta_path)
    header_to_idx = {name.strip(): idx for idx, name in enumerate(header)}
    audio_idx = header_to_idx["audio_file"]

    for cols in iter_metadata_rows(meta_path):
        if len(cols) <= audio_idx:
            continue
        wav_path = resolve_dataset_wav(cols[audio_idx], dataset_dir)
        return str(wav_path)
    raise RuntimeError(f"No usable rows in metadata: {meta_path}")



def _copy_if_needed(src: Path | None, dst: Path):
    if not src or not src.exists():
        return
    src_resolved = src.resolve()
    dst_resolved = dst.resolve() if dst.exists() else dst
    if src_resolved == dst_resolved:
        return
    import shutil

    shutil.copy2(src, dst)



def resolve_base_model_files(custom_base_dir: Path | None):
    dvae_link = "https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth"
    mel_link = "https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth"
    vocab_link = "https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json"
    model_link = "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth"

    dvae_path = CHECKPOINTS_OUT_PATH / "dvae.pth"
    mel_path = CHECKPOINTS_OUT_PATH / "mel_stats.pth"
    vocab_path = CHECKPOINTS_OUT_PATH / "vocab.json"
    model_path = CHECKPOINTS_OUT_PATH / "model.pth"

    if custom_base_dir and custom_base_dir.exists():
        custom_dvae = first_existing(custom_base_dir / "dvae.pth")
        custom_mel = first_existing(custom_base_dir / "mel_stats.pth")
        custom_vocab = first_existing(custom_base_dir / "vocab.json")
        custom_model = first_existing(custom_base_dir / "model.pth", custom_base_dir / "best_model.pth")

        _copy_if_needed(custom_vocab, vocab_path)
        _copy_if_needed(custom_model, model_path)
        _copy_if_needed(custom_dvae, dvae_path)
        _copy_if_needed(custom_mel, mel_path)

    missing = []
    if not dvae_path.exists():
        missing.append(dvae_link)
    if not mel_path.exists():
        missing.append(mel_link)
    if not vocab_path.exists():
        missing.append(vocab_link)
    if not model_path.exists():
        missing.append(model_link)

    if missing:
        print("Downloading missing XTTS base files ...")
        ModelManager._download_model_files(missing, str(CHECKPOINTS_OUT_PATH), progress_bar=True)

    return model_path, vocab_path, dvae_path, mel_path



def apply_speed_config(config, trainer_args) -> None:
    set_attr_if_present(config, "mixed_precision", USE_AMP)
    set_attr_if_present(config, "precision", AMP_PRECISION if USE_AMP else "float32")
    set_attr_if_present(config, "num_loader_workers", NUM_LOADER_WORKERS)
    set_attr_if_present(config, "pin_memory", PIN_MEMORY)
    set_attr_if_present(config, "persistent_workers", PERSISTENT_WORKERS and NUM_LOADER_WORKERS > 0)
    if NUM_LOADER_WORKERS > 0:
        set_attr_if_present(config, "prefetch_factor", PREFETCH_FACTOR)

    # Best-effort flags requested from CLI.
    if USE_AUGMENTATION and not set_attr_if_present(config, "use_augmentation", True):
        print("Warning: this trainer/config version has no 'use_augmentation' field; flag accepted but not applied.")
    if MONITOR_GRADIENTS:
        applied = False
        for attr_name in ("monitor_gradients", "print_grad_norm", "log_grad_norm", "model_param_stats"):
            applied = set_attr_if_present(config, attr_name, True) or applied
        if not applied:
            print("Warning: this trainer/config version has no gradient monitoring field; flag accepted but not applied.")

    set_attr_if_present(trainer_args, "mixed_precision", USE_AMP)
    set_attr_if_present(trainer_args, "precision", AMP_PRECISION if USE_AMP else "float32")

    print("Applied trainer speed config:")
    for name in [
        "mixed_precision",
        "precision",
        "num_loader_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "use_augmentation",
        "monitor_gradients",
        "print_grad_norm",
        "log_grad_norm",
        "model_param_stats",
    ]:
        if hasattr(config, name):
            print(f"  config.{name} = {getattr(config, name)}")
    for name in ["mixed_precision", "precision"]:
        if hasattr(trainer_args, name):
            print(f"  trainer_args.{name} = {getattr(trainer_args, name)}")



def maybe_compile_model(model):
    if not USE_TORCH_COMPILE:
        print("torch.compile disabled. Use --compile_model to enable it.")
        return model

    if not hasattr(torch, "compile"):
        print("torch.compile is not available in this torch build.")
        return model

    compile_target = model
    target_name = "model"

    if hasattr(model, "xtts") and hasattr(model.xtts, "gpt"):
        compile_target = model.xtts.gpt
        target_name = "model.xtts.gpt"

    try:
        compiled_target = torch.compile(
            compile_target,
            backend=TORCH_COMPILE_BACKEND,
            mode=TORCH_COMPILE_MODE,
            fullgraph=False,
            dynamic=True,
        )
        if target_name == "model":
            model = compiled_target
        else:
            model.xtts.gpt = compiled_target
        print(
            f"Enabled torch.compile on {target_name} "
            f"(backend={TORCH_COMPILE_BACKEND}, mode={TORCH_COMPILE_MODE})."
        )
    except Exception as exc:
        warnings.warn(
            "torch.compile could not be enabled cleanly. "
            f"Continuing without it. Details: {exc}"
        )

    return model



def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    apply_cli_args(args)
    print_launch_config(args)
    enable_runtime_speedups()

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
    if not TRAIN_META.exists():
        raise FileNotFoundError(f"Missing train metadata: {TRAIN_META}")
    if not EVAL_META.exists():
        raise FileNotFoundError(f"Missing eval metadata: {EVAL_META}")

    validate_coqui_metadata(TRAIN_META, DATASET_DIR)
    validate_coqui_metadata(EVAL_META, DATASET_DIR)

    custom_base_dir = PROJECT_ROOT / "checkpoints" / "XTTS_base"
    xtts_checkpoint, tokenizer_file, dvae_checkpoint, mel_norm_file = resolve_base_model_files(custom_base_dir)

    speaker_reference = [parse_reference_wav(EVAL_META if EVAL_META.exists() else TRAIN_META, DATASET_DIR)]

    config_dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="belarusian_custom",
        path=str(DATASET_DIR),
        meta_file_train=TRAIN_META.name,
        meta_file_val=EVAL_META.name,
        language=LANGUAGE,
    )

    datasets = [config_dataset]

    model_args = GPTArgs(
        max_conditioning_length=132300,
        min_conditioning_length=66150,
        debug_loading_failures=False,
        max_wav_length=255995,
        max_text_length=200,
        mel_norm_file=str(mel_norm_file),
        dvae_checkpoint=str(dvae_checkpoint),
        xtts_checkpoint=str(xtts_checkpoint),
        tokenizer_file=str(tokenizer_file),
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    audio_config = XttsAudioConfig(
        sample_rate=22050,
        dvae_sample_rate=22050,
        output_sample_rate=24000,
    )

    config = GPTTrainerConfig(
        output_path=str(OUT_PATH),
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="Belarusian GPT XTTS training on coqui-ai-TTS (Coqui formatter)",
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=max(1, BATCH_SIZE),
        num_loader_workers=NUM_LOADER_WORKERS,
        eval_split_max_size=256,
        epochs=NUM_EPOCHS,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=SAVE_STEP,
        save_n_checkpoints=3,
        save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=LEARNING_RATE,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[
            {
                "text": TEST_SENTENCE,
                "speaker_wav": speaker_reference,
                "language": LANGUAGE,
            }
        ],
        languages=[LANGUAGE],
    )

    trainer_args = TrainerArgs(
        restore_path=None,
        skip_train_epoch=False,
        start_with_eval=START_WITH_EVAL,
        grad_accum_steps=GRAD_ACCUM_STEPS,
    )
    apply_speed_config(config, trainer_args)

    model = GPTTrainer.init_from_config(config)
    model = maybe_compile_model(model)

    train_samples, eval_samples = load_tts_samples(
        datasets,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    trainer = Trainer(
        trainer_args,
        config,
        output_path=str(OUT_PATH),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
