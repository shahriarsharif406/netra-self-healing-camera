"""NETRA agent: observe -> plan -> (approve) -> act -> verify -> keep or roll back.

Design rules
  * The planner can only pick tools from a fixed registry (netra.fixes.CANDIDATES).
  * Every step passes an approval gate. Low-risk tools may run automatically, medium-risk
    tools (geometry changes, regional edits) need a human by default. No approver = denied.
  * Every step is verified by re-running the diagnosis; a step that does not clearly help,
    or that makes the frame worse overall, is rolled back and the next tool is tried.
  * The original frame is never modified. Everything is written to a JSONL audit log.
"""
from __future__ import annotations
import hashlib
import json
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import numpy as np

from .diagnose import Diagnosis, Thresholds, diagnose, saturation_mask, sharpness_index, _gray
from .fixes import CANDIDATES, TOOLS

IMPROVE_FACTOR = 0.7      # 'applied': target severity reduced to <= 70 %
PARTIAL_FACTOR = 0.85     # 'partial': reduced to <= 85 %, kept but the fault stays open
MAX_NET_REGRESSION = 0.05 # ... without raising the total severity of the frame by more
MAX_ATTEMPTS = 2          # per fault type per run

ALERTS = {
    "drift": "Camera has moved ({detail}). Check the mount and re-validate any fixed regions of interest.",
    "lens_dirt": "Smudge or dirt covers about {pct:.0f}% of the view. Schedule a lens cleaning; the software fix is only a mitigation.",
    "low_light": "Scene is much darker than at commissioning. Check lighting, exposure and gain settings.",
    "overexposure": "Scene is brighter than at commissioning with clipped highlights. Check exposure, glare and IR/lighting.",
    "noise": "Sensor noise is high. Check gain/ISO, lighting and cabling.",
    "blur": "Image is out of focus or smeared. Check focus, lens condition and vibration.",
}


class AuditLog:
    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else None
        self.events: list[dict] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _clean(o):
        if isinstance(o, dict):
            return {str(k): AuditLog._clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [AuditLog._clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return f"<ndarray {o.shape}>"
        return o

    def write(self, event: str, run_id: str, **data):
        rec = {"ts": round(time.time(), 3), "run_id": run_id, "event": event, **self._clean(data)}
        self.events.append(rec)
        if self.path:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")


# ------------------------------------------------------------------ planners ---------
class RulePlanner:
    """Deterministic, offline. Picks the highest-priority unresolved fault."""
    name = "rules"

    def next_fault(self, diag: Diagnosis, attempts: Counter, blocked: set, history: list) -> Optional[str]:
        for f in diag.findings:        # already sorted by FAULT_PRIORITY
            if f.fault not in blocked and attempts[f.fault] < MAX_ATTEMPTS:
                return f.fault
        return None

    def tool_order(self, fault: str, diag: Diagnosis, history: list) -> list[str]:
        return list(CANDIDATES[fault])


class LLMPlanner(RulePlanner):
    """Optional: let a language model choose the fault and the tool order.

    `llm_call(prompt) -> str` can wrap any provider (Bedrock, Anthropic, Gemini ...).
    The model only *proposes*: its answer is validated against the live diagnosis and the
    fixed tool registry, anything else falls back to the rule planner, and the approval gate
    and verification still apply to whatever it proposes.
    """
    name = "llm"

    def __init__(self, llm_call: Callable[[str], str]):
        self.llm_call = llm_call
        self.fallbacks = 0
        self._last: dict = {}

    def _ask(self, diag, history) -> dict:
        prompt = (
            "You control a camera-repair agent. Choose which detected fault to fix next and the order "
            "of tools to try. Reply with JSON only: "
            '{"fault": <name>, "tool_order": [<tool>...], "reason": <short>}.\n'
            f"Detected faults: {json.dumps(AuditLog._clean(diag.to_dict()['findings']))}\n"
            f"Allowed tools per fault: {json.dumps(CANDIDATES)}\n"
            f"Steps so far: {json.dumps(AuditLog._clean(history[-4:]))}\n"
            "Fix geometry (drift) before anything else; denoise before deblurring."
        )
        try:
            raw = self.llm_call(prompt)
            return json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        except Exception:
            return {}

    def next_fault(self, diag, attempts, blocked, history):
        ans = self._ask(diag, history)
        fault = ans.get("fault")
        valid = [f.fault for f in diag.findings if f.fault not in blocked and attempts[f.fault] < MAX_ATTEMPTS]
        if fault in valid and isinstance(ans.get("tool_order"), list):
            self._last = ans
            return fault
        self._last = {}
        if valid:
            self.fallbacks += 1
        return super().next_fault(diag, attempts, blocked, history)

    def tool_order(self, fault, diag, history):
        allowed = CANDIDATES[fault]
        proposed = [t for t in self._last.get("tool_order", []) if t in allowed]
        rest = [t for t in allowed if t not in proposed]
        return proposed + rest if proposed else list(allowed)


# ------------------------------------------------------------------ result -----------
@dataclass
class RunResult:
    run_id: str
    original: np.ndarray
    output: np.ndarray
    initial: Diagnosis
    final: Diagnosis
    steps: list[dict]
    alerts: list[str]
    unresolved: list[str]
    runtime_ms: float
    planner: str

    def summary(self) -> dict:
        return {"run_id": self.run_id, "planner": self.planner,
                "initial_faults": self.initial.faults(), "final_faults": self.final.faults(),
                "steps": [{k: s[k] for k in ("fault", "tool", "status", "decision", "before", "after")}
                          for s in self.steps],
                "unresolved": self.unresolved, "runtime_ms": round(self.runtime_ms, 1)}


# ------------------------------------------------------------------ agent ------------
class Agent:
    """policy: "auto_low" (default) | "ask_all" | "auto_all"."""

    def __init__(self, reference: np.ndarray, thresholds: Optional[Thresholds] = None,
                 planner: Optional[RulePlanner] = None, policy: str = "auto_low",
                 log_path: Optional[str] = None, max_steps: int = 8, time_budget_s: float = 20.0,
                 task: Optional[Callable[[np.ndarray], float]] = None, task_tolerance: float = 0.0,
                 passport=None):
        """task: optional downstream score (higher = better, e.g. fraction of QR codes decoded in the user's
        regions). A repair that lowers it by more than task_tolerance is rolled back, even if the image
        quality metrics improved. passport: a netra.passport.Passport that records every run."""
        assert policy in ("auto_low", "ask_all", "auto_all")
        self.ref = reference
        self.th = thresholds or Thresholds()
        self.planner = planner or RulePlanner()
        self.policy = policy
        self.max_steps = max_steps
        self.time_budget_s = time_budget_s
        self.log = AuditLog(log_path)
        self.task, self.task_tol, self.passport = task, task_tolerance, passport
        self._task_cur = None
        g = _gray(reference)
        self._ref_stats = {"ref": reference, "ref_mean": float(g.mean()), "ref_sharp": sharpness_index(g)}

    # --- approval gate ---------------------------------------------------------------
    def _decide(self, tool, fault, approver) -> tuple[bool, str]:
        if self.policy == "auto_all":
            return True, "auto (policy=auto_all)"
        if self.policy == "auto_low" and tool.risk == "low":
            return True, "auto (low risk)"
        if approver is None:
            return False, "denied: needs human approval and no approver is connected"
        ok = bool(approver({"fault": fault, "tool": tool.name, "risk": tool.risk,
                            "description": tool.description}))
        return ok, "human approved" if ok else "human denied"

    # --- verification ----------------------------------------------------------------
    @staticmethod
    def _verify(fault: str, before: Diagnosis, after: Diagnosis) -> tuple[str, str]:
        """Returns ("applied" | "partial" | "rolled_back", reason)."""
        tb, ta = before.severity(fault), after.severity(fault)
        improved = ta == 0 or ta <= IMPROVE_FACTOR * tb
        partial = ta <= PARTIAL_FACTOR * tb
        # a latent smudge can only be judged once global faults are gone; do not blame the fix
        excl = () if "lens_dirt" in before.faults() or fault == "lens_dirt" else ("lens_dirt",)
        net = after.total_severity(excl) - before.total_severity(excl)
        new = [f for f in after.faults() if f not in before.faults() and f not in excl]
        if not (improved or partial):
            return "rolled_back", f"target severity {tb:.2f} -> {ta:.2f} (needs <= {PARTIAL_FACTOR * tb:.2f})"
        if net > MAX_NET_REGRESSION:
            return "rolled_back", f"frame got worse overall (+{net:.2f}); new faults: {new}"
        status = "applied" if improved else "partial"
        return status, f"target severity {tb:.2f} -> {ta:.2f}; net change {net:+.2f}; new faults: {new}"

    # --- one fault -------------------------------------------------------------------
    def _handle(self, fault, cur, diag, approver, run_id, history):
        ctx = dict(self._ref_stats, drift_M=diag.artifacts.get("drift_M"),
                   dirt_mask=diag.artifacts.get("dirt_mask"), dirt_alpha=diag.artifacts.get("dirt_alpha"))
        for tool_name in self.planner.tool_order(fault, diag, history):
            tool = TOOLS[tool_name]
            ok, why = self._decide(tool, fault, approver)
            rec = {"fault": fault, "tool": tool_name, "risk": tool.risk, "decision": why,
                   "before": round(diag.severity(fault), 3), "after": None, "params": {}, "note": "",
                   "latency_ms": 0.0}
            if not ok:
                rec["status"] = "denied"
                self.log.write("step", run_id, **rec)
                history.append(rec)
                return None, None, rec
            t0 = time.perf_counter()
            try:
                new, params = tool.fn(cur, ctx)
                after = diagnose(new, self.ref, self.th, self._excl)
            except Exception as e:                       # tool failure must never crash the loop
                rec.update(status="failed", note=f"{type(e).__name__}: {e}",
                           latency_ms=round((time.perf_counter() - t0) * 1000, 1))
                self.log.write("step", run_id, **rec)
                history.append(rec)
                continue
            status, reason = self._verify(fault, diag, after)
            if status != "rolled_back" and self.task is not None:
                try:
                    ts = float(self.task(new))
                except Exception:
                    ts = None
                if ts is not None and self._task_cur is not None and ts < self._task_cur - self.task_tol:
                    status, reason = "rolled_back", (f"task guard: downstream score {self._task_cur:.2f} -> "
                                                     f"{ts:.2f}; image metrics alone said keep ({reason})")
                    rec["task_guard"] = True
                elif ts is not None:
                    rec["task_score"] = round(ts, 3)
                    self._task_next = ts
            good = status != "rolled_back"
            rec.update(params=params, after=round(after.severity(fault), 3), note=reason, status=status,
                       latency_ms=round((time.perf_counter() - t0) * 1000, 1))
            self.log.write("step", run_id, **rec)
            history.append(rec)
            if good:
                return new, after, rec
        return None, None, history[-1]

    # --- main loop -------------------------------------------------------------------
    def run(self, frame: np.ndarray, approver: Optional[Callable[[dict], bool]] = None,
            t_days: Optional[float] = None) -> RunResult:
        t0 = time.perf_counter()
        run_id = uuid.uuid4().hex[:10]
        cur = frame.copy()
        self._excl = saturation_mask(frame)    # from the ORIGINAL frame: clipped areas stay unjudgeable
        diag0 = diagnose(cur, self.ref, self.th, self._excl)
        diag = diag0
        self._task_cur = float(self.task(cur)) if self.task is not None else None
        self._task_next = None
        self.log.write("run_start", run_id, frame_sha1=hashlib.sha1(frame.tobytes()[:65536]).hexdigest()[:12],
                       policy=self.policy, planner=self.planner.name, diagnosis=diag0.to_dict())
        attempts, blocked, history = Counter(), set(), []
        for _ in range(self.max_steps):
            if time.perf_counter() - t0 > self.time_budget_s:
                self.log.write("budget_exceeded", run_id, seconds=round(time.perf_counter() - t0, 1))
                break
            fault = self.planner.next_fault(diag, attempts, blocked, history)
            if fault is None:
                break
            attempts[fault] += 1
            new, after, rec = self._handle(fault, cur, diag, approver, run_id, history)
            if new is None:
                blocked.add(fault)
            else:
                cur, diag = new, after
                if self.task is not None and getattr(self, "_task_next", None) is not None:
                    self._task_cur = self._task_next
        final = diagnose(cur, self.ref, self.th, self._excl)
        alerts = self._alerts(diag0) + self._late_alerts(diag0, history)
        unresolved = sorted(set(final.faults()) | blocked)
        ms = (time.perf_counter() - t0) * 1000
        self.log.write("run_end", run_id, final=final.to_dict(), unresolved=unresolved,
                       alerts=alerts, runtime_ms=round(ms, 1))
        forecasts = []
        if self.passport is not None and t_days is not None:
            self.passport.record(t_days, diag0.metrics, diag0.faults(), note=run_id)
            forecasts = self.passport.report(t_days)
            for f in forecasts:
                self.log.write("forecast", run_id, **{k: v for k, v in f.items()})
        res = RunResult(run_id, frame, cur, diag0, final, history, alerts, unresolved, ms, self.planner.name)
        res.forecasts = forecasts
        return res

    @staticmethod
    def _late_alerts(diag0: Diagnosis, history: list) -> list[str]:
        """Faults that only showed up after other repairs are reported with lower confidence:
        residual repair artefacts can mimic them, so a human should look before acting."""
        out, seen = [], set(diag0.faults())
        for s in history:
            if s["fault"] not in seen:
                seen.add(s["fault"])
                text = ALERTS[s["fault"]].format(pct=100 * s["before"] / 4, detail="see audit log")
                out.append(f"[lower confidence: found after other repairs, verify on site] {text}")
        return out

    @staticmethod
    def _alerts(diag: Diagnosis) -> list[str]:
        out = []
        for f in diag.findings:
            if f.fault == "drift":
                d = f.detail
                out.append(ALERTS["drift"].format(detail=f"{d['drift_px']} px, {d['drift_deg']} deg, scale {d['drift_scale']}"))
            elif f.fault == "lens_dirt":
                out.append(ALERTS["lens_dirt"].format(pct=100 * f.detail["dirt_fraction"]))
            else:
                out.append(ALERTS[f.fault])
        return out
