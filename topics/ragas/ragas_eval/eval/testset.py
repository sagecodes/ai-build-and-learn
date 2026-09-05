import json
import os

from config import DATA_DIR

TESTSET_PATH = DATA_DIR / "testset.json"


def load_testset() -> list[dict]:
    if not TESTSET_PATH.exists():
        raise FileNotFoundError(
            f"Testset not found at {TESTSET_PATH}. "
            "Run generate_testset.py first:\n"
            "  docker exec -w /app rag-comparison-ragas "
            "python ragas_eval/generate_testset.py"
        )
    with open(TESTSET_PATH) as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Testset at {TESTSET_PATH} must be a non-empty list of dicts.")
    bad = [i for i, r in enumerate(data) if "question" not in r or "ground_truth" not in r]
    if bad:
        raise ValueError(
            f"Testset rows missing 'question' or 'ground_truth': indices {bad}"
        )
    return data


def save_testset(records: list[dict]) -> None:
    """Atomic write — partial writes never corrupt the testset."""
    tmp = TESTSET_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(records, f, indent=2)
    os.replace(tmp, TESTSET_PATH)
