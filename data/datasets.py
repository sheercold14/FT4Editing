from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    # Allows lightweight scripts (e.g., suite generation) to run without torch.
    class Dataset:  # type: ignore
        pass


def _read_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_records(path: str) -> List[Dict]:
    return _read_jsonl(path) if path.endswith(".jsonl") else _read_json(path)


@dataclass
class RecordMapper:
    prompt_key_candidates: tuple = ("prompt", "src")
    target_key_candidates: tuple = ("target_new", "alt")
    rephrase_key_candidates: tuple = ("rephrase_prompt", "rephrase")
    negative_key_candidates: tuple = ("target_old", "pred", "ground_truth")

    def pick(self, record: Dict, keys: tuple, default: str = "") -> str:
        for key in keys:
            if key in record and record[key] is not None:
                value = str(record[key]).strip()
                if value:
                    return value
        return default

    def canonicalize(self, record: Dict) -> Dict:
        prompt = self.pick(record, self.prompt_key_candidates)
        target = self.pick(record, self.target_key_candidates)
        rephrase = self.pick(record, self.rephrase_key_candidates)
        rejected = self.pick(record, self.negative_key_candidates)
        return {
            **record,
            "prompt": prompt,
            "target_new": target,
            "rephrase_prompt": rephrase,
            "rejected": rejected,
        }


class OffPolicySFTDataset(Dataset):
    def __init__(self, path: str, mapper: Optional[RecordMapper] = None):
        mapper = mapper or RecordMapper()
        self.records = [mapper.canonicalize(rec) for rec in load_records(path)]
        self.records = [r for r in self.records if r["prompt"] and r["target_new"]]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict:
        item = self.records[index]
        return {
            "prompt": item["prompt"],
            "target": item["target_new"],
            "record": item,
        }


class OnPolicySFTDataset(Dataset):
    def __init__(self, path: str, min_conflict: float = 0.0):
        self.records = load_records(path)
        filtered = []
        for record in self.records:
            score = float(record.get("conflict_score", 0.0))
            if score >= min_conflict and record.get("prompt") and record.get("chosen"):
                filtered.append(record)
        self.records = filtered

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict:
        item = self.records[index]
        return {
            "prompt": item["prompt"],
            "target": item["chosen"],
            "rejected": item.get("rejected", ""),
            "conflict_score": float(item.get("conflict_score", 0.0)),
            "record": item,
        }


class PreferenceDataset(Dataset):
    def __init__(self, path: str):
        self.records = load_records(path)
        self.records = [
            r for r in self.records if r.get("prompt") and r.get("chosen") and r.get("rejected")
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict:
        record = self.records[index]
        return {
            "prompt": record["prompt"],
            "chosen": record["chosen"],
            "rejected": record["rejected"],
            "conflict_score": float(record.get("conflict_score", 0.0)),
            "record": record,
        }


def dump_jsonl(path: str, rows: List[Dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
