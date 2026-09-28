"""Shared NLI (contradiction) model used by the graph, MIS and sampleMIS defenses.

The upstream code hard-coded the original author's Princeton path
(/scratch/gpfs/zs7353/DeBERTa-v3-large-mnli-fever-anli-ling-wanli), which does
not exist on Wulver. This loads the same public checkpoint from the Hugging
Face Hub instead.

Override with NLI_MODEL_PATH (an HF repo id or a local directory). Downloads go
to HF_HOME; scripts/wulver_run_one.slurm points that at a shared cache under
/project so the model is fetched once (see scripts/prefetch_nli.sh) and compute
nodes never need internet access.
"""
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
NLI_MODEL = os.environ.get("NLI_MODEL_PATH", DEFAULT_NLI_MODEL)


def load_nli(model_path=None, device=None):
    """Return (tokenizer, model, contradiction_label_index)."""
    model_path = model_path or NLI_MODEL
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    # Look the index up from the config instead of assuming 2, so a different
    # NLI checkpoint with another label order can't silently flip the graph.
    label2id = {str(k).lower(): int(v) for k, v in model.config.label2id.items()}
    if "contradiction" not in label2id:
        raise ValueError(f"NLI model {model_path} has no 'contradiction' label: {model.config.label2id}")
    return tokenizer, model, label2id["contradiction"]
