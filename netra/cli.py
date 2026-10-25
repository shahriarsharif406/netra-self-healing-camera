"""Command line:  python -m netra.cli --ref ref.png --frame cam.png --out fixed.png [--yes-risky]"""
import argparse
import json
import cv2
from .agent import Agent


def main():
    ap = argparse.ArgumentParser(description="NETRA self-healing camera agent")
    ap.add_argument("--ref", required=True, help="reference frame from the healthy camera")
    ap.add_argument("--frame", required=True, help="current frame to diagnose and repair")
    ap.add_argument("--out", default="netra_fixed.png")
    ap.add_argument("--log", default="netra_audit.jsonl")
    ap.add_argument("--yes-risky", action="store_true",
                    help="pre-approve medium-risk fixes (re-alignment, smudge correction)")
    a = ap.parse_args()
    ref, frame = cv2.imread(a.ref), cv2.imread(a.frame)
    if ref is None or frame is None:
        raise SystemExit("could not read input images")
    frame = cv2.resize(frame, (ref.shape[1], ref.shape[0]))
    agent = Agent(ref, policy="auto_low", log_path=a.log)
    res = agent.run(frame, approver=(lambda s: True) if a.yes_risky else None)
    cv2.imwrite(a.out, res.output)
    print(json.dumps(res.summary(), indent=2))
    for al in res.alerts:
        print("ALERT:", al)


if __name__ == "__main__":
    main()
