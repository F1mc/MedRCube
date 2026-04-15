import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


# Default task split mirrors benchmark_postanalysis/analysis_shortcut.py
DEFAULT_HIGH_LEVEL_TASKS = {
    "Disease Diagnosis",
    "Abnormity Diagnosis",
    "Severity Grading",
}

DEFAULT_IGNORE_TASKS = {
    "Report Generation",
    "View Identification",
}


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


@dataclass(frozen=True)
class ShortcutReport:
    Group_A: int
    Group_B: int
    Group_C: int
    Group_D: int
    Total_Pairs: int
    Luck_Rate: float
    Shortcut_Prob: float
    Rationality: float


def compute_shortcut_metrics(
    records: List[Dict[str, Any]],
    *,
    valid_ids: Optional[Set[str]] = None,
    high_level_tasks: Optional[Set[str]] = None,
    ignore_tasks: Optional[Set[str]] = None,
) -> ShortcutReport:
    """
    Compute shortcut/luck metrics from a flat `results.json` list.

    Required per record:
      - id: str
      - image_path: str (same image groups low/high tasks)
      - task: str
      - correct: bool

    Logic follows benchmark_postanalysis/analysis_shortcut.py:
      Pairwise compare every (high_id, low_id) per image.
    """
    high_level_tasks = high_level_tasks or set(DEFAULT_HIGH_LEVEL_TASKS)
    ignore_tasks = ignore_tasks or set(DEFAULT_IGNORE_TASKS)

    # Build image -> {high:[id], low:[id]} from the base records
    image_map: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {"high": [], "low": []})
    id_to_rec: Dict[str, Dict[str, Any]] = {}

    for r in records:
        rid = _safe_str(r.get("id"))
        if not rid:
            continue
        if valid_ids is not None and rid not in valid_ids:
            continue
        id_to_rec[rid] = r

        task = r.get("task")
        if task in ignore_tasks:
            continue
        img_key = _safe_str(r.get("image_path"))
        if not img_key:
            continue
        if task in high_level_tasks:
            image_map[img_key]["high"].append(rid)
        else:
            image_map[img_key]["low"].append(rid)

    target_images = [img for img, d in image_map.items() if d["high"] and d["low"]]

    stats = {"Group_A": 0, "Group_B": 0, "Group_C": 0, "Group_D": 0}

    for img in target_images:
        highs = image_map[img]["high"]
        lows = image_map[img]["low"]

        for h_id in highs:
            h_rec = id_to_rec.get(h_id)
            if not h_rec:
                continue
            h_res = bool(h_rec.get("correct")) is True
            for l_id in lows:
                l_rec = id_to_rec.get(l_id)
                if not l_rec:
                    continue
                l_res = bool(l_rec.get("correct")) is True

                if l_res and h_res:
                    stats["Group_A"] += 1
                elif l_res and not h_res:
                    stats["Group_B"] += 1
                elif (not l_res) and (not h_res):
                    stats["Group_C"] += 1
                elif (not l_res) and h_res:
                    stats["Group_D"] += 1

    total = sum(stats.values())
    if total == 0:
        return ShortcutReport(
            Group_A=0,
            Group_B=0,
            Group_C=0,
            Group_D=0,
            Total_Pairs=0,
            Luck_Rate=0.0,
            Shortcut_Prob=0.0,
            Rationality=0.0,
        )

    high_correct_total = stats["Group_A"] + stats["Group_D"]
    luck_rate = (stats["Group_D"] / high_correct_total) if high_correct_total > 0 else 0.0

    low_wrong_total = stats["Group_C"] + stats["Group_D"]
    shortcut_prob = (stats["Group_D"] / low_wrong_total) if low_wrong_total > 0 else 0.0

    rationality = 1.0 - luck_rate

    return ShortcutReport(
        Group_A=stats["Group_A"],
        Group_B=stats["Group_B"],
        Group_C=stats["Group_C"],
        Group_D=stats["Group_D"],
        Total_Pairs=total,
        Luck_Rate=luck_rate,
        Shortcut_Prob=shortcut_prob,
        Rationality=rationality,
    )


def main():
    import argparse
    import csv

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True, help="Path to results.json (flat list).")
    parser.add_argument(
        "--valid_ids_json",
        type=str,
        default=None,
        help="Optional JSON list of valid IDs (co-occurrence filtering).",
    )
    parser.add_argument(
        "--high_level_tasks_json",
        type=str,
        default=None,
        help="Optional JSON list of high-level task names.",
    )
    parser.add_argument(
        "--ignore_tasks_json",
        type=str,
        default=None,
        help="Optional JSON list of ignored task names.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Output CSV path (default: alongside input).",
    )
    args = parser.parse_args()

    records = _load_json(args.results)
    if not isinstance(records, list):
        raise ValueError("results.json must be a list of records")

    valid_ids = set(map(str, _load_json(args.valid_ids_json))) if args.valid_ids_json else None
    high_level_tasks = set(map(str, _load_json(args.high_level_tasks_json))) if args.high_level_tasks_json else None
    ignore_tasks = set(map(str, _load_json(args.ignore_tasks_json))) if args.ignore_tasks_json else None

    rep = compute_shortcut_metrics(
        records,
        valid_ids=valid_ids,
        high_level_tasks=high_level_tasks,
        ignore_tasks=ignore_tasks,
    )

    out_csv = args.out_csv or args.results.replace("results.json", "analysis_hierarchy_metrics.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "Model",
                "Group_A",
                "Group_B",
                "Group_C",
                "Group_D",
                "Total_Pairs",
                "Luck_Rate",
                "Shortcut_Prob",
                "Rationality",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "Model": "single_model",
                "Group_A": rep.Group_A,
                "Group_B": rep.Group_B,
                "Group_C": rep.Group_C,
                "Group_D": rep.Group_D,
                "Total_Pairs": rep.Total_Pairs,
                "Luck_Rate": rep.Luck_Rate,
                "Shortcut_Prob": rep.Shortcut_Prob,
                "Rationality": rep.Rationality,
            }
        )

    print(
        json.dumps(
            {
                "Group_A": rep.Group_A,
                "Group_B": rep.Group_B,
                "Group_C": rep.Group_C,
                "Group_D": rep.Group_D,
                "Total_Pairs": rep.Total_Pairs,
                "Luck_Rate": rep.Luck_Rate,
                "Shortcut_Prob": rep.Shortcut_Prob,
                "Rationality": rep.Rationality,
                "out_csv": out_csv,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

