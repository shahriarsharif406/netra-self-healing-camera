"""NETRA web UI.  Run:  streamlit run app.py"""
import json
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from netra import scenes, metrics
from netra.agent import Agent
from netra.diagnose import diagnose
from netra.faults import FAULTS, apply_faults
from netra.fixes import CANDIDATES, TOOLS

st.set_page_config(page_title="NETRA - self-healing camera agent", layout="wide")
MAX_W = 800


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_upload(f):
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.error("Could not read that image.")
        st.stop()
    s = min(1.0, MAX_W / img.shape[1])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


@st.cache_data(show_spinner=False)
def demo_pair(name, seed):
    ref, live, pl = scenes.make_pair(name, np.random.default_rng(seed))
    return ref, live, pl


st.title("NETRA - a camera agent that diagnoses and repairs its own feed")
st.caption("OpenCV 5 | diagnose -> plan -> approve -> fix -> verify -> roll back | every step is logged")

with st.sidebar:
    st.header("1. Input")
    mode = st.radio("Source", ["Demo (simulated faults)", "Upload my own frames"])
    truth = None
    if mode.startswith("Demo"):
        scene = st.selectbox("Scene", scenes.TEST_SCENES + scenes.CALIB_SCENES)
        seed = st.number_input("Random seed", 0, 9999, 1)
        chosen = st.multiselect("Faults to inject", FAULTS, default=["drift", "noise"])
        sev = st.slider("Severity", 0.1, 1.0, 0.7)
        ref, live, pl = demo_pair(scene, int(seed))
        frame = apply_faults(live, chosen, np.random.default_rng(int(seed)), sev=sev)[0] if chosen else live
        truth = live
    else:
        up_ref = st.file_uploader("Reference frame (camera when it was healthy)", type=["png", "jpg", "jpeg"])
        up_cur = st.file_uploader("Current frame", type=["png", "jpg", "jpeg"])
        if not (up_ref and up_cur):
            st.info("Upload both frames, or switch to the demo.")
            st.stop()
        ref = load_upload(up_ref)
        frame = cv2.resize(load_upload(up_cur), (ref.shape[1], ref.shape[0]))
    st.header("2. What may the agent do?")
    st.caption("Low-risk fixes (brightness, noise, sharpness) run automatically. Risky ones need your yes.")
    ok_geo = st.checkbox("Allow re-aligning the image (changes geometry)", value=False)
    ok_reg = st.checkbox("Allow editing smudged regions", value=False)

c1, c2 = st.columns(2)
c1.subheader("Reference (healthy)")
c1.image(rgb(ref), width="stretch")
c2.subheader("Current frame")
c2.image(rgb(frame), width="stretch")

diag = diagnose(frame, ref)
st.subheader("Diagnosis")
if not diag.findings:
    st.success("No fault found relative to the reference. Nothing to do.")
else:
    rows = []
    for f in diag.findings:
        t = TOOLS[CANDIDATES[f.fault][0]]
        rows.append({"fault": f.fault, "severity (0-1)": round(f.severity, 2), "evidence": json.dumps(f.detail),
                     "first tool": t.name, "risk": t.risk,
                     "needs your approval": "yes" if t.risk != "low" else "no"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

if st.button("Repair", type="primary", disabled=not diag.findings):
    allowed = {"align_orb": ok_geo, "align_ecc": ok_geo, "dehaze_local": ok_reg, "inpaint_telea": ok_reg,
               "inpaint_ns": ok_reg}
    agent = Agent(ref, policy="auto_low")
    with st.spinner("Working..."):
        res = agent.run(frame, approver=lambda s: allowed.get(s["tool"], False))
    st.subheader("Result")
    a, b = st.columns(2)
    a.image(rgb(res.original), caption="Before", width="stretch")
    b.image(rgb(res.output), caption="After", width="stretch")
    if truth is not None:
        m1, m2, m3 = st.columns(3)
        m1.metric("SSIM vs ground truth (demo only)", f"{metrics.ssim(truth, res.output):.3f}",
                  f"{metrics.ssim(truth, res.output) - metrics.ssim(truth, frame):+.3f}")
        m2.metric("PSNR dB", f"{metrics.psnr(truth, res.output):.1f}",
                  f"{metrics.psnr(truth, res.output) - metrics.psnr(truth, frame):+.1f}")
        m3.metric("Agent time", f"{res.runtime_ms:.0f} ms")
    st.subheader("What the agent did")
    if res.steps:
        st.dataframe(pd.DataFrame([{k: s[k] for k in ("fault", "tool", "risk", "decision", "status", "note", "latency_ms")}
                                   for s in res.steps]), hide_index=True, width="stretch")
    if res.unresolved:
        st.warning("Needs a human: " + ", ".join(res.unresolved))
    st.subheader("Maintenance alerts")
    for al in res.alerts:
        st.write("- " + al)
    st.download_button("Download audit log (JSONL)", "\n".join(json.dumps(e) for e in agent.log.events),
                       file_name=f"netra_{res.run_id}.jsonl")
    ok, buf = cv2.imencode(".png", res.output)
    st.download_button("Download repaired frame", buf.tobytes(), file_name="netra_repaired.png", mime="image/png")


# ---------------------------------------------------------------- Health Passport ---------
st.divider()
st.subheader("Health Passport: when will this camera need maintenance?")
st.caption("Simulated: one frame per day for a camera that slowly degrades. Each day's diagnosis goes into the "
           "camera's passport; a Theil-Sen trend with a Mann-Kendall test forecasts when a fault limit is crossed. "
           "Forecasts are only raised for statistically significant trends.")
from netra import timeline
from netra.passport import Passport, WATCHES


@st.cache_data(show_spinner="Simulating 50 days of camera history...")
def passport_history(scene, process, seed):
    pp = Passport(f"sim-{process}")
    for d, ref, fr in timeline.frames(scene, process, 50, seed):
        dg = diagnose(fr, ref)
        pp.record(d, dg.metrics, dg.faults())
    return pp.records


pc1, pc2, pc3 = st.columns(3)
p_scene = pc1.selectbox("Scene", scenes.FRESH_SCENES[:4], key="p_scene")
p_proc = pc2.selectbox("Slow degradation", timeline.PROCESSES, index=1, key="p_proc")
p_day = pc3.slider("Today is day", 15, 49, 30, key="p_day")
recs = passport_history(p_scene, p_proc, 7)
pp = Passport("sim")
pp.records = [r for r in recs]
alerts = pp.report(float(p_day), horizon_days=30)
if alerts:
    for f in alerts:
        st.warning(f["message"] + f"  (trend p={f['p']}, window {f.get('window_days')} d)")
else:
    st.success("No fault is on course to cross its limit within 30 days.")
rows = []
for w in WATCHES:
    t, y = pp.series(w.metric, until=float(p_day))
    if len(t):
        rows.append(pd.DataFrame({"day": t, w.metric: y}).set_index("day"))
if rows:
    st.line_chart(pd.concat(rows, axis=1))
st.dataframe(pd.DataFrame([pp.forecast(w, float(p_day)) for w in WATCHES])[
    ["fault", "metric", "threshold", "status", "eta_days", "slope_per_day", "p"]], hide_index=True)
