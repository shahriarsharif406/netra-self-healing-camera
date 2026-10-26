"""Run: python -m pytest -q   (or python tests/test_netra.py)"""
import numpy as np
import pytest
import cv2

from netra import scenes, faults, fixes, metrics
from netra.agent import Agent, LLMPlanner
from netra.diagnose import diagnose


@pytest.fixture(scope="module")
def pair():
    rng = np.random.default_rng(7)
    return scenes.make_pair("astronaut", rng) + (rng,)


def test_runs_on_opencv_5():
    assert int(cv2.__version__.split(".")[0]) >= 5


def test_clean_frame_has_no_findings(pair):
    ref, live, _, _ = pair
    assert diagnose(live, ref).faults() == []


@pytest.mark.parametrize("fault", faults.FAULTS)
def test_each_fault_is_detected(pair, fault):
    ref, live, _, rng = pair
    bad, _ = faults.apply_faults(live, [fault], rng, sev=0.8)
    assert fault in diagnose(bad, ref).faults()


def test_agent_repairs_drift_and_noise(pair):
    ref, live, _, rng = pair
    bad, _ = faults.apply_faults(live, ["drift", "noise"], rng, sev=0.7)
    res = Agent(ref, policy="auto_all").run(bad)
    assert metrics.ssim(live, res.output) > metrics.ssim(live, bad) + 0.3
    assert not res.unresolved


def test_original_frame_is_never_modified(pair):
    ref, live, _, rng = pair
    bad, _ = faults.apply_faults(live, ["blur"], rng, sev=0.8)
    snap = bad.copy()
    Agent(ref, policy="auto_all").run(bad)
    assert np.array_equal(bad, snap)


def test_medium_risk_step_is_denied_without_approver(pair):
    ref, live, _, rng = pair
    bad, _ = faults.apply_faults(live, ["drift"], rng, sev=0.8)
    res = Agent(ref, policy="auto_low").run(bad)
    assert [s["status"] for s in res.steps] == ["denied"]
    assert np.array_equal(res.output, bad)
    assert "drift" in res.unresolved


def test_human_can_approve_and_deny(pair):
    ref, live, _, rng = pair
    bad, _ = faults.apply_faults(live, ["drift"], rng, sev=0.8)
    asked = []
    ok = Agent(ref, policy="auto_low").run(bad, approver=lambda s: asked.append(s) or True)
    assert asked and asked[0]["tool"] == "align_orb" and ok.steps[0]["status"] == "applied"
    no = Agent(ref, policy="auto_low").run(bad, approver=lambda s: False)
    assert no.steps[0]["status"] == "denied"


def test_low_risk_runs_automatically(pair):
    ref, live, _, rng = pair
    bad, _ = faults.apply_faults(live, ["noise"], rng, sev=0.8)
    res = Agent(ref, policy="auto_low").run(bad)     # no approver needed
    assert res.steps[0]["decision"].startswith("auto") and res.steps[0]["status"] != "denied"


def test_failing_tool_does_not_crash_agent(pair, monkeypatch):
    ref, live, _, rng = pair
    bad, _ = faults.apply_faults(live, ["noise"], rng, sev=0.8)

    def boom(img, ctx):
        raise RuntimeError("simulated tool crash")
    monkeypatch.setitem(fixes.TOOLS, "denoise_bilateral",
                        fixes.Tool("denoise_bilateral", "noise", "low", boom, "broken"))
    res = Agent(ref, policy="auto_all").run(bad)
    assert res.steps[0]["status"] == "failed"
    assert any(s["tool"] == "denoise_nlm" and s["status"] in ("applied", "partial") for s in res.steps)


def test_harmful_fix_is_rolled_back(pair, monkeypatch):
    ref, live, _, rng = pair
    bad, _ = faults.apply_faults(live, ["noise"], rng, sev=0.8)
    noop = lambda img, ctx: (img.copy(), {})
    for name in ("denoise_bilateral", "denoise_nlm"):
        monkeypatch.setitem(fixes.TOOLS, name, fixes.Tool(name, "noise", "low", noop, "does nothing"))
    res = Agent(ref, policy="auto_all").run(bad)
    assert all(s["status"] == "rolled_back" for s in res.steps)
    assert np.array_equal(res.output, bad) and "noise" in res.unresolved


def test_llm_planner_cannot_invent_tools(pair):
    ref, live, _, rng = pair
    bad, _ = faults.apply_faults(live, ["noise"], rng, sev=0.8)
    evil = lambda prompt: '{"fault": "noise", "tool_order": ["rm_rf", "denoise_nlm"], "reason": "x"}'
    res = Agent(ref, planner=LLMPlanner(evil), policy="auto_all").run(bad)
    assert all(s["tool"] in fixes.TOOLS for s in res.steps) and res.steps[0]["tool"] == "denoise_nlm"


def test_llm_planner_falls_back_on_garbage(pair):
    ref, live, _, rng = pair
    bad, _ = faults.apply_faults(live, ["noise"], rng, sev=0.8)
    p = LLMPlanner(lambda prompt: "I think you should try turning it off and on again")
    res = Agent(ref, planner=p, policy="auto_all").run(bad)
    assert p.fallbacks >= 1 and res.steps[0]["status"] != "failed"


def test_audit_log_is_valid_jsonl(pair, tmp_path):
    import json
    ref, live, _, rng = pair
    bad, _ = faults.apply_faults(live, ["low_light"], rng, sev=0.8)
    log = tmp_path / "audit.jsonl"
    Agent(ref, policy="auto_all", log_path=str(log)).run(bad)
    events = [json.loads(l) for l in log.read_text().splitlines()]
    assert events[0]["event"] == "run_start" and events[-1]["event"] == "run_end"
    assert any(e["event"] == "step" for e in events)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ---------------- Health Passport -------------------------------------------------------
from netra.passport import Passport, WATCHES, mann_kendall, theil_sen


def _drift_passport(slope, n=30, noise=0.05, seed=0):
    r = np.random.default_rng(seed)
    pp = Passport("t")
    for d in range(n):
        pp.record(d, {"drift_px": 0.2 + slope * d + r.normal(0, noise)}, [])
    return pp


def test_theil_sen_and_mk():
    t = np.arange(20.0)
    s, i = theil_sen(t, 2 * t + 1)
    assert abs(s - 2) < 1e-9 and abs(i - 1) < 1e-9
    tau, p = mann_kendall(2 * t)
    assert tau == 1.0 and p < 1e-3
    assert mann_kendall(np.random.default_rng(1).normal(size=40))[1] > 0.01


def test_forecast_eta_for_creeping_drift():
    pp = _drift_passport(0.1)                       # crosses 3.0 px at day ~28
    w = [w for w in WATCHES if w.fault == "drift"][0]
    f = pp.forecast(w, 20.0)
    assert f["status"] == "degrading"
    assert abs(20 + f["eta_days"] - 28) < 3


def test_no_forecast_for_flat_camera():
    pp = _drift_passport(0.0)
    assert pp.report(29.0) == []


def test_insufficient_data():
    pp = _drift_passport(0.1, n=5)
    assert pp.forecast([w for w in WATCHES if w.fault == "drift"][0], 4.0)["status"] == "insufficient_data"


# ---------------- Downstream task guard ---------------------------------------------------
def test_task_guard_rolls_back_harmful_repair(pair):
    ref, live, _, rng = pair
    bad, _ = faults.apply_faults(live, ["noise"], rng, sev=0.8)
    calls = {"n": 0}

    def task(img):                                    # task score that punishes any change to the frame
        calls["n"] += 1
        return 1.0 if np.array_equal(img, bad) else 0.0
    res = Agent(ref, policy="auto_all", task=task).run(bad)
    assert any(s.get("task_guard") for s in res.steps)
    assert np.array_equal(res.output, bad)            # every repair vetoed, original content kept


def test_agent_records_into_passport(pair):
    ref, live, _, _ = pair
    pp = Passport("cam-x")
    Agent(ref, passport=pp).run(live, t_days=3.0)
    assert len(pp.records) == 1 and pp.records[0]["t"] == 3.0


# ---------------- Provenance ---------------------------------------------------------------
from netra import provenance


def test_audit_chain_detects_tampering(pair):
    ref, live, _, rng = pair
    bad, _ = faults.apply_faults(live, ["noise"], rng, sev=0.8)
    ag = Agent(ref)
    res = ag.run(bad)
    ev = [dict(e) for e in ag.log.events]
    assert provenance.verify_chain(ev) == (True, None)
    assert provenance.verify_receipt(res.receipt, ev, res.output) == []
    ev[1]["tool"] = "forged"                        # edit one past entry
    ok, idx = provenance.verify_chain(ev)
    assert not ok and idx == 1
    del ev[2]                                       # deletion is caught too
    assert not provenance.verify_chain(ev)[0]


def test_receipt_detects_swapped_image(pair):
    ref, live, _, rng = pair
    bad, _ = faults.apply_faults(live, ["low_light"], rng, sev=0.8)
    ag = Agent(ref)
    res = ag.run(bad)
    assert res.receipt["modified"]
    other = res.output.copy(); other[0, 0] = 255 - other[0, 0]
    assert "repaired image does not match the receipt" in provenance.verify_receipt(res.receipt, ag.log.events, other)


# ---------------- Fleet + report ---------------------------------------------------------------
from netra import fleet as fleetmod
from netra.report_html import build_report


def _fake_cam(name, slope, n=40):
    r = np.random.default_rng(3)
    recs = [{"camera": name, "t": float(d), "faults": [], "note": "",
             "metrics": {"drift_px": 0.2 + slope * d + float(r.normal(0, .03))}} for d in range(n)]
    return {"camera": name, "scene": "x", "process": "test", "records": recs}


def test_fleet_ranking_orders_by_urgency():
    fl = [_fake_cam("flat", 0.0), _fake_cam("slow", 0.06), _fake_cam("past", 0.2)]
    rows = fleetmod.rank(fl, 35.0)
    assert [r["camera"] for r in rows][0] == "past" and rows[0]["status"] == "needs attention now"
    assert rows[-1]["camera"] == "flat" and rows[-1]["status"] == "healthy"
    assert rows[1]["status"] == "maintenance soon" and rows[1]["eta_days"] > 0
    assert rows[0]["health"] < rows[-1]["health"]


def test_html_report_contains_provenance(pair):
    ref, live, _, rng = pair
    bad, _ = faults.apply_faults(live, ["noise"], rng, sev=0.8)
    ag = Agent(ref)
    res = ag.run(bad)
    ok, _b = provenance.verify_chain(ag.log.events)
    doc = build_report(res, res.receipt, ok, len(ag.log.events), camera="<script>x</script>")
    assert res.receipt["repaired_sha256"] in doc and "Audit chain verified" in doc
    assert "<script>x</script>" not in doc          # camera name is escaped
