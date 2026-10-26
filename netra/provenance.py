"""Tamper-evident audit trail and repair receipts.

Every audit event carries the SHA-256 of the previous event, so editing, deleting or reordering any
past entry breaks the chain from that point on. A repair receipt binds the original frame, the repaired
frame and the chain head together, so a repaired image can be shown to differ from the original and
to come from a logged run. This is integrity evidence, not a signature: someone who can rewrite the
whole log AND the receipt can forge both. Sign the head hash (or anchor it elsewhere) for stronger proof.

CLI:  python -m netra.provenance audit.jsonl [receipt.json [repaired.png]]
"""
from __future__ import annotations
import hashlib
import json
import sys
import numpy as np

GENESIS = "0" * 64


def event_hash(prev: str, rec: dict) -> str:
    body = {k: v for k, v in rec.items() if k != "hash"}
    body["prev"] = prev
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_chain(events: list[dict]) -> tuple[bool, int | None]:
    """Returns (ok, index_of_first_bad_event)."""
    prev = GENESIS
    for i, e in enumerate(events):
        if e.get("prev") != prev or e.get("hash") != event_hash(prev, e):
            return False, i
        prev = e["hash"]
    return True, None


def image_sha256(img: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(img).tobytes() + str(img.shape).encode()).hexdigest()


def make_receipt(run_id: str, original: np.ndarray, output: np.ndarray, events: list[dict], steps: list[dict]) -> dict:
    mine = [e for e in events if e.get("run_id") == run_id]
    return {"run_id": run_id,
            "original_sha256": image_sha256(original),
            "repaired_sha256": image_sha256(output),
            "modified": not np.array_equal(original, output),
            "chain_head": events[-1]["hash"] if events else GENESIS,
            "run_events": len(mine),
            "applied_steps": [{"fault": s["fault"], "tool": s["tool"], "status": s["status"]}
                              for s in steps if s.get("status") in ("applied", "partial")],
            "notice": "Repaired frame. Not original evidence; see audit log for every modification."}


def verify_receipt(receipt: dict, events: list[dict], repaired: np.ndarray | None = None) -> list[str]:
    """Returns a list of problems (empty = consistent)."""
    problems = []
    ok, bad = verify_chain(events)
    if not ok:
        problems.append(f"audit chain broken at event {bad}")
    hashes = {e.get("hash") for e in events}
    if receipt.get("chain_head") not in hashes and events:
        problems.append("receipt chain head is not in this log")
    if not any(e.get("run_id") == receipt.get("run_id") for e in events):
        problems.append("run id not found in log")
    if repaired is not None and image_sha256(repaired) != receipt.get("repaired_sha256"):
        problems.append("repaired image does not match the receipt")
    return problems


if __name__ == "__main__":
    import cv2
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ev = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
    ok, bad = verify_chain(ev)
    print(f"audit chain: {'OK' if ok else f'BROKEN at event {bad}'} ({len(ev)} events)")
    if len(sys.argv) > 2:
        rc = json.load(open(sys.argv[2]))
        img = cv2.imread(sys.argv[3]) if len(sys.argv) > 3 else None
        pr = verify_receipt(rc, ev, img)
        print("receipt:", "consistent" if not pr else "; ".join(pr))
        sys.exit(0 if ok and not pr else 1)
    sys.exit(0 if ok else 1)
