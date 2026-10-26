"""NETRA web UI.  Run:  streamlit run app.py"""
import base64
import html
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from netra import scenes, metrics, timeline, provenance, fleet as fleetmod
from netra.report_html import build_report
from netra.agent import Agent
from netra.diagnose import diagnose
from netra.faults import FAULTS, apply_faults
from netra.fixes import CANDIDATES, TOOLS
from netra.passport import Passport, WATCHES

st.set_page_config(page_title="NETRA | Self-healing camera agent", page_icon="👁", layout="wide")
MAX_W = 800

# ------------------------------------------------------------------ design system ----------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--bg:#070b14;--card:#0e1626;--card2:#121c30;--line:#1d2a44;--txt:#e6edf7;--mut:#8da0bf;
--cy:#22d3ee;--vi:#8b5cf6;--ok:#34d399;--warn:#fbbf24;--bad:#f87171;}
html,body,[class*="css"]{font-family:'Inter',system-ui,sans-serif;}
.stApp{background:radial-gradient(1200px 600px at 12% -10%,#12204a55 0,transparent 60%),
 radial-gradient(900px 500px at 100% 0,#3b1d6a44 0,transparent 55%),var(--bg);}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1.4rem;max-width:1280px;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1222,#080d19);border-right:1px solid var(--line);}
h1,h2,h3{letter-spacing:-.02em;}
.hero{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:22px;padding:34px 38px;margin-bottom:18px;
 background:linear-gradient(135deg,#0d1730 0%,#14224a 55%,#2a1b57 100%);}
.hero:before{content:"";position:absolute;right:-60px;top:-60px;width:340px;height:340px;border-radius:50%;
 background:radial-gradient(circle,#22d3ee55,transparent 65%);}
.hero:after{content:"";position:absolute;right:70px;top:48px;width:120px;height:120px;border-radius:50%;
 border:2px solid #22d3ee88;box-shadow:0 0 0 14px #22d3ee14,0 0 0 30px #22d3ee0c,inset 0 0 40px #22d3ee33;}
.eyebrow{font:500 12px 'JetBrains Mono',monospace;letter-spacing:.18em;color:var(--cy);text-transform:uppercase;}
.hero h1{font-size:46px;font-weight:800;margin:6px 0 8px;line-height:1.05;
 background:linear-gradient(90deg,#fff,#a5f3fc 60%,#c4b5fd);-webkit-background-clip:text;background-clip:text;color:transparent;}
.hero p{color:#b7c4dd;font-size:16px;max-width:640px;margin:0 0 16px;}
.pill{display:inline-block;padding:4px 11px;margin:0 6px 6px 0;border-radius:999px;font:500 12px 'Inter';
 border:1px solid #2c3d63;background:#0e1a33aa;color:#b7c4dd;}
.pill.cy{border-color:#22d3ee66;color:#67e8f9}.pill.ok{border-color:#34d39966;color:#6ee7b7}
.pill.warn{border-color:#fbbf2466;color:#fcd34d}.pill.bad{border-color:#f8717166;color:#fca5a5}
.pill.vi{border-color:#8b5cf666;color:#c4b5fd}
.card{background:linear-gradient(180deg,var(--card2),var(--card));border:1px solid var(--line);border-radius:16px;padding:18px 20px;}
.kpi{background:linear-gradient(180deg,var(--card2),var(--card));border:1px solid var(--line);border-radius:16px;padding:16px 18px;}
.kpi .l{color:var(--mut);font:500 11px 'JetBrains Mono';letter-spacing:.12em;text-transform:uppercase}
.kpi .v{font-size:32px;font-weight:700;margin-top:2px}
.kpi .d{font-size:13px;margin-top:2px}.up{color:var(--ok)}.dn{color:var(--bad)}.fl{color:var(--mut)}
.sec{font:600 12px 'JetBrains Mono';letter-spacing:.16em;color:var(--mut);text-transform:uppercase;margin:22px 0 8px;}
.fault{display:flex;align-items:center;gap:14px;padding:12px 14px;border:1px solid var(--line);border-radius:12px;
 background:#0b1426;margin-bottom:8px;}
.fault .n{width:130px;font-weight:600}
.bar{flex:1;height:8px;border-radius:99px;background:#16233c;overflow:hidden}
.bar>i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,var(--cy),var(--vi))}
.fault .s{width:44px;text-align:right;font:500 13px 'JetBrains Mono';color:#cbd5e1}
.step{border-left:2px solid var(--line);margin-left:8px;padding:2px 0 16px 20px;position:relative}
.step:before{content:"";position:absolute;left:-7px;top:6px;width:12px;height:12px;border-radius:50%;background:var(--c,#64748b);
 box-shadow:0 0 0 4px #0b1426}
.step .t{font-weight:600}.step .m{color:var(--mut);font-size:13px;margin-top:2px}
.step code{font:12px 'JetBrains Mono';color:#a5b4fc}
.alert{border:1px solid #fbbf2455;background:#fbbf2410;border-radius:12px;padding:12px 14px;margin-bottom:8px;color:#fde68a}
.alert.ok{border-color:#34d39955;background:#34d39910;color:#a7f3d0}
.foot{color:var(--mut);font-size:12px;text-align:center;margin:34px 0 8px}
div[data-testid="stTabs"] button[role="tab"]{font-weight:600;font-size:15px}
.stButton>button[kind="primary"]{background:linear-gradient(90deg,#06b6d4,#8b5cf6);border:0;color:white;font-weight:700;
 border-radius:12px;padding:.65rem 1.4rem;box-shadow:0 8px 28px #22d3ee33}
.stButton>button[kind="primary"]:hover{filter:brightness(1.1)}
.stDownloadButton>button{border-radius:12px;border:1px solid var(--line);background:var(--card)}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers ----------------
def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def b64(img_bgr):
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode()


def load_upload(f):
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.error("Could not read that image.")
        st.stop()
    s = min(1.0, MAX_W / img.shape[1])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


@st.cache_data(show_spinner=False)
def demo_pair(name, seed):
    return scenes.make_pair(name, np.random.default_rng(seed))


def compare_slider(before, after, height=480):
    """Before/after image comparison slider (pure HTML/JS, no extra dependency)."""
    h, w = before.shape[:2]
    doc = f"""
<div id="w" style="position:relative;width:100%;max-width:{w}px;margin:auto;aspect-ratio:{w}/{h};border-radius:16px;
overflow:hidden;border:1px solid #1d2a44;user-select:none;touch-action:none;font-family:Inter,sans-serif">
<img src="data:image/jpeg;base64,{b64(after)}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover">
<div id="top" style="position:absolute;inset:0;width:50%;overflow:hidden">
<img id="bi" src="data:image/jpeg;base64,{b64(before)}" style="position:absolute;left:0;top:0;height:100%;object-fit:cover">
</div>
<div id="bar" style="position:absolute;top:0;bottom:0;left:50%;width:3px;background:#22d3ee;box-shadow:0 0 18px #22d3ee"></div>
<div id="knob" style="position:absolute;top:50%;left:50%;width:42px;height:42px;margin:-21px 0 0 -21px;border-radius:50%;
background:#0b1426;border:2px solid #22d3ee;color:#22d3ee;display:flex;align-items:center;justify-content:center;font-weight:700">&#8596;</div>
<span style="position:absolute;left:12px;top:12px;background:#000a;color:#fca5a5;padding:4px 10px;border-radius:99px;font-size:12px">BEFORE</span>
<span style="position:absolute;right:12px;top:12px;background:#000a;color:#6ee7b7;padding:4px 10px;border-radius:99px;font-size:12px">AFTER</span>
</div>
<script>
const w=document.getElementById('w'),t=document.getElementById('top'),b=document.getElementById('bar'),k=document.getElementById('knob'),bi=document.getElementById('bi');
function fit(){{bi.style.width=w.offsetWidth+'px'}} fit(); addEventListener('resize',fit);
function set(x){{const r=w.getBoundingClientRect();let p=Math.min(1,Math.max(0,(x-r.left)/r.width))*100;
t.style.width=p+'%';b.style.left=p+'%';k.style.left=p+'%';}}
let d=false;w.addEventListener('pointerdown',e=>{{d=true;set(e.clientX)}});
addEventListener('pointerup',()=>d=false);addEventListener('pointermove',e=>{{if(d)set(e.clientX)}});
</script>"""
    components.html(doc, height=height)


def kpi(label, value, delta=None, good=True):
    d = ""
    if delta is not None:
        cls = "fl" if delta == "" else ("up" if good else "dn")
        d = f'<div class="d {cls}">{html.escape(delta)}</div>'
    return f'<div class="kpi"><div class="l">{html.escape(label)}</div><div class="v">{html.escape(str(value))}</div>{d}</div>'


STATUS_COLOR = {"applied": "#34d399", "partial": "#fbbf24", "rolled_back": "#f87171", "denied": "#f59e0b", "failed": "#f87171"}


# ------------------------------------------------------------------ hero -------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">Self-healing vision infrastructure</div>
  <h1>NETRA</h1>
  <p>A camera agent that works out <b>why</b> its own feed is degraded, repairs it with OpenCV 5,
  <b>verifies</b> the repair, rolls back what did not help, and asks a human before anything risky.</p>
  <span class="pill cy">OpenCV 5.0</span><span class="pill vi">Diagnose &rarr; Plan &rarr; Approve &rarr; Fix &rarr; Verify</span>
  <span class="pill ok">Human-in-the-loop</span><span class="pill">Full audit log</span><span class="pill warn">Runs on CPU, ~650 ms/frame</span>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ sidebar ----------------
with st.sidebar:
    st.markdown("### 1 &nbsp; Input")
    mode = st.radio("Source", ["Demo (simulated faults)", "Upload my own frames", "Live webcam (real camera)"],
                    label_visibility="collapsed")
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
        if mode.startswith("Live"):
            st.caption("Real camera test: 1) shoot a healthy reference, 2) then degrade the camera for real "
                       "(cover part of the lens with a finger smudge, dim the room, shake the laptop, add distance) "
                       "and shoot again.")
            up_ref = st.camera_input("1 · Reference (healthy)", key="cam_ref")
            up_cur = st.camera_input("2 · Current (after degrading)", key="cam_cur") if up_ref else None
        else:
            up_ref = st.file_uploader("Reference frame (camera when it was healthy)", type=["png", "jpg", "jpeg"])
            up_cur = st.file_uploader("Current frame", type=["png", "jpg", "jpeg"])
        if not (up_ref and up_cur):
            st.info("Provide both frames (upload or webcam), or switch to the demo.")
            st.stop()
        ref = load_upload(up_ref)
        frame = cv2.resize(load_upload(up_cur), (ref.shape[1], ref.shape[0]))
    st.markdown("### 2 &nbsp; Agent permissions")
    st.caption("Low-risk fixes (brightness, noise, sharpness) run automatically. Risky ones need your yes. "
               "With no approval ticked, they are denied.")
    ok_geo = st.checkbox("Allow re-aligning the image (changes geometry)", value=False)
    ok_reg = st.checkbox("Allow editing smudged regions", value=False)

tab_repair, tab_pass, tab_fleet, tab_proof = st.tabs(
    ["🛠  Diagnose & repair", "📈  Health Passport", "🛰  Fleet", "🏆  Measured results"])

# ================================================================= TAB 1: repair ==========
with tab_repair:
    c1, c2 = st.columns(2, gap="medium")
    c1.markdown('<div class="sec">Reference · healthy camera</div>', unsafe_allow_html=True)
    c1.image(rgb(ref), width="stretch")
    c2.markdown('<div class="sec">Current frame · live feed</div>', unsafe_allow_html=True)
    c2.image(rgb(frame), width="stretch")

    diag = diagnose(frame, ref)
    st.markdown('<div class="sec">Diagnosis</div>', unsafe_allow_html=True)
    if not diag.findings:
        st.markdown('<div class="alert ok">No fault found relative to the reference. Nothing to do.</div>',
                    unsafe_allow_html=True)
    else:
        for f in diag.findings:
            t = TOOLS[CANDIDATES[f.fault][0]]
            risk_cls = "ok" if t.risk == "low" else "warn"
            need = "auto" if t.risk == "low" else "needs approval"
            st.markdown(
                f'<div class="fault"><div class="n">{f.fault.replace("_", " ")}</div>'
                f'<div class="bar"><i style="width:{max(3, int(f.severity * 100))}%"></i></div>'
                f'<div class="s">{f.severity:.2f}</div>'
                f'<span class="pill cy">{t.name}</span><span class="pill {risk_cls}">{t.risk} risk · {need}</span></div>',
                unsafe_allow_html=True)
        with st.expander("Evidence (raw measurements)"):
            st.json({f.fault: f.detail for f in diag.findings})

    run = st.button("⚡  Repair this frame", type="primary", disabled=not diag.findings)
    if run:
        allowed = {"align_orb": ok_geo, "align_ecc": ok_geo, "dehaze_local": ok_reg, "inpaint_telea": ok_reg,
                   "inpaint_ns": ok_reg}
        agent = Agent(ref, policy="auto_low")
        with st.spinner("Agent is working..."):
            res = agent.run(frame, approver=lambda s: allowed.get(s["tool"], False))
        st.session_state["res"] = (res, [dict(e) for e in agent.log.events], truth is not None)

    if "res" in st.session_state and diag.findings:
        res, events, has_truth = st.session_state["res"]
        st.markdown('<div class="sec">Result · drag the slider</div>', unsafe_allow_html=True)
        compare_slider(res.original, res.output)
        if has_truth and truth is not None and res.original.shape == truth.shape:
            s1, s0 = metrics.ssim(truth, res.output), metrics.ssim(truth, res.original)
            p1, p0 = metrics.psnr(truth, res.output), metrics.psnr(truth, res.original)
            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(kpi("SSIM vs ground truth", f"{s1:.3f}", f"{s1 - s0:+.3f} vs faulty", s1 >= s0), unsafe_allow_html=True)
            k2.markdown(kpi("PSNR (dB)", f"{p1:.1f}", f"{p1 - p0:+.1f} dB", p1 >= p0), unsafe_allow_html=True)
            k3.markdown(kpi("Faults left", len(res.final.faults()), f"from {len(res.initial.faults())}",
                            len(res.final.faults()) <= len(res.initial.faults())), unsafe_allow_html=True)
            k4.markdown(kpi("Agent time", f"{res.runtime_ms:.0f} ms", ""), unsafe_allow_html=True)
            st.caption("Ground truth is only known in the demo; the agent never sees it.")

        st.markdown('<div class="sec">What the agent did</div>', unsafe_allow_html=True)
        if res.steps:
            for s in res.steps:
                col = STATUS_COLOR.get(s["status"], "#64748b")
                st.markdown(
                    f'<div class="step" style="--c:{col}"><div class="t">{s["fault"].replace("_", " ")} &rarr; '
                    f'<code>{s["tool"]}</code> &nbsp;<span class="pill" style="color:{col};border-color:{col}88">'
                    f'{s["status"].replace("_", " ")}</span></div>'
                    f'<div class="m">{html.escape(str(s["decision"]))} · {s["latency_ms"]} ms</div>'
                    f'<div class="m">{html.escape(str(s["note"]))}</div></div>', unsafe_allow_html=True)
        if res.unresolved:
            st.markdown('<div class="alert">👤 Needs a human: ' + ", ".join(res.unresolved) + "</div>",
                        unsafe_allow_html=True)
        if res.alerts:
            st.markdown('<div class="sec">Maintenance alerts</div>', unsafe_allow_html=True)
            for al in res.alerts:
                st.markdown(f'<div class="alert">🔧 {html.escape(al)}</div>', unsafe_allow_html=True)
        chain_ok, _bad = provenance.verify_chain(events)
        problems = provenance.verify_receipt(res.receipt, events, res.output)
        st.markdown('<div class="sec">Provenance · tamper-evident record</div>', unsafe_allow_html=True)
        if chain_ok and not problems:
            st.markdown(f'<div class="alert ok">🔒 Audit chain verified ({len(events)} hash-linked events). '
                        f'Repaired frame <code>{res.receipt["repaired_sha256"][:16]}…</code> is bound to this run. '
                        'Editing or deleting any past entry breaks the chain.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert">⚠ Provenance check failed: ' + html.escape("; ".join(problems)) + "</div>",
                        unsafe_allow_html=True)
        d1, d2, d3, d4 = st.columns(4)
        d4.download_button("⬇ Health report (HTML)",
                           build_report(res, res.receipt, chain_ok, len(events), getattr(res, "forecasts", []),
                                        camera="demo camera"),
                           file_name=f"netra_report_{res.run_id}.html", mime="text/html")
        d1.download_button("⬇ Audit log (JSONL)", "\n".join(json.dumps(e) for e in events),
                           file_name=f"netra_{res.run_id}.jsonl")
        d3.download_button("⬇ Repair receipt (JSON)", json.dumps(res.receipt, indent=2),
                           file_name=f"netra_receipt_{res.run_id}.json")
        ok, buf = cv2.imencode(".png", res.output)
        d2.download_button("⬇ Repaired frame", buf.tobytes(), file_name="netra_repaired.png", mime="image/png")

# ================================================================= TAB 2: passport ========
with tab_pass:
    st.markdown("#### Camera Health Passport")
    st.caption("Simulated: one frame per day for a camera that slowly degrades. Each day's diagnosis goes into the "
               "camera's passport. A Theil-Sen trend with a Mann-Kendall test forecasts when a fault limit will be "
               "crossed, and only for statistically significant trends.")

    @st.cache_data(show_spinner="Simulating 50 days of camera history...")
    def passport_history(scene_, process, seed_):
        pp_ = Passport(f"sim-{process}")
        for d, ref_, fr in timeline.frames(scene_, process, 50, seed_):
            dg = diagnose(fr, ref_)
            pp_.record(d, dg.metrics, dg.faults())
        return pp_.records

    pc1, pc2, pc3 = st.columns(3)
    p_scene = pc1.selectbox("Scene", scenes.FRESH_SCENES[:4], key="p_scene")
    p_proc = pc2.selectbox("Slow degradation", timeline.PROCESSES, index=1, key="p_proc")
    p_day = pc3.slider("Today is day", 15, 49, 30, key="p_day")
    pp = Passport("camera-01")
    pp.records = passport_history(p_scene, p_proc, 7)
    now = float(p_day)
    alerts = pp.report(now, horizon_days=30)
    if alerts:
        for f in alerts:
            st.markdown(f'<div class="alert">⏳ <b>{f["fault"].replace("_", " ")}</b> — '
                        f'{html.escape(f["message"])} <span class="pill">p = {f["p"]}</span></div>',
                        unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert ok">✅ No fault is on course to cross its limit within 30 days.</div>',
                    unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3), facecolor="#0e1626")
    show = [("drift_px", "Drift (px)"), ("noise_excess", "Noise excess"), ("sharp_ratio", "Sharpness ratio")]
    for ax, (metric, title) in zip(axes, show):
        t, y = pp.series(metric, until=now)
        ax.set_facecolor("#0e1626")
        ax.plot(t, y, color="#22d3ee", lw=2)
        w = [w for w in WATCHES if w.metric == metric][0]
        ax.axhline(w.threshold, color="#f87171", ls="--", lw=1.2)
        ax.axvline(now, color="#8b5cf6", lw=1, alpha=.6)
        fc = pp.forecast(w, now)
        if fc["status"] == "degrading" and fc["slope_per_day"] is not None and len(t):
            xs = np.array([now, now + min(fc["eta_days"], 30)])
            ys = fc["value_now"] + fc["slope_per_day"] * (xs - now)
            ax.plot(xs, ys, color="#fbbf24", ls=":", lw=2)
        ax.set_title(title, color="#e6edf7", fontsize=11, loc="left")
        ax.tick_params(colors="#8da0bf", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#1d2a44")
        ax.grid(color="#1d2a44", lw=.6)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption("Cyan: measured daily. Red dashed: fault limit. Amber dotted: forecast to the limit.")
    rows = pd.DataFrame([pp.forecast(w, now) for w in WATCHES])[
        ["fault", "metric", "threshold", "status", "eta_days", "slope_per_day", "p"]]
    st.dataframe(rows, hide_index=True, width="stretch")

# ================================================================= TAB 3: fleet ===========
with tab_fleet:
    st.markdown("#### Fleet maintenance queue")
    st.caption("Six SIMULATED cameras, each with its own slow degradation. NETRA ranks them by urgency using each "
               "camera's Health Passport: already past a limit first, then fewest days to the forecast limit. "
               "The ranking logic is real; the cameras and their faults are simulated.")
    if not fleetmod.CACHE.exists():
        st.info("Fleet cache not built yet. Run `python -m netra.fleet --build` (about 3 minutes), then reload.")
    else:
        fl = fleetmod.load()
        f_day = st.slider("Today is day", 15, 49, 35, key="f_day")
        rows_ = fleetmod.rank(fl, float(f_day))
        n_now = sum(r["status"] == "needs attention now" for r in rows_)
        n_soon = sum(r["status"] == "maintenance soon" for r in rows_)
        k = st.columns(3)
        k[0].markdown(kpi("Cameras", len(rows_), ""), unsafe_allow_html=True)
        k[1].markdown(kpi("Need attention now", n_now, "", False), unsafe_allow_html=True)
        k[2].markdown(kpi("Maintenance soon (30 d)", n_soon, "", False), unsafe_allow_html=True)
        st.markdown('<div class="sec">Maintenance queue · most urgent first</div>', unsafe_allow_html=True)
        cls = {"needs attention now": "bad", "maintenance soon": "warn", "healthy": "ok"}
        for i, r in enumerate(rows_, 1):
            what = "" if r["fault"] is None else r["fault"].replace("_", " ")
            eta = "" if r["eta_days"] is None else ("limit reached" if r["eta_days"] == 0 else f"limit in ~{r['eta_days']:.0f} d")
            st.markdown(
                f'<div class="fault"><div class="n">#{i} {r["camera"]}</div>'
                f'<div class="bar"><i style="width:{int(r["health"])}%"></i></div>'
                f'<div class="s">{int(r["health"])}</div>'
                f'<span class="pill {cls[r["status"]]}">{r["status"]}</span>'
                + (f'<span class="pill">{what} {eta}</span>' if what else "") + "</div>", unsafe_allow_html=True)
        st.caption("Bar = health score (100 = every metric at its baseline, 0 = a fault limit reached). "
                   "Dirt build-up forecasts are the weakest (see README).")

# ================================================================= TAB 4: results =========
with tab_proof:
    st.markdown("#### Measured on fresh scenes the tuning never saw")
    st.caption("192 faulty and 16 clean frames from 8 fresh collage scenes. Faults are simulated. "
               "Ground truth is known to the evaluator, never to the agent.")
    k = st.columns(4)
    k[0].markdown(kpi("Mean SSIM", "0.843", "from 0.498 faulty", True), unsafe_allow_html=True)
    k[1].markdown(kpi("Frames made worse", "0%", "naive enhance: 71%", True), unsafe_allow_html=True)
    k[2].markdown(kpi("False alarms (clean)", "0 / 16", "", True), unsafe_allow_html=True)
    k[3].markdown(kpi("Time per frame", "~650 ms", "2 CPU cores, no GPU", True), unsafe_allow_html=True)
    k = st.columns(4)
    k[0].markdown(kpi("Task guard: QR score", "0.79", "from 0.72 without guard", True), unsafe_allow_html=True)
    k[1].markdown(kpi("Frames hurting the task", "0", "without guard: 17", True), unsafe_allow_html=True)
    k[2].markdown(kpi("Forecast error, day 30", "1-5 d", "drift / noise / focus / dimming", True), unsafe_allow_html=True)
    k[3].markdown(kpi("Stable-camera false alerts", "0 / 24", "", True), unsafe_allow_html=True)
    base = Path(__file__).parent / "results" / "final_fresh" / "figures"
    g1, g2 = st.columns(2)
    if (base / "fig_ssim_by_fault.png").exists():
        g1.image(str(base / "fig_ssim_by_fault.png"), caption="SSIM by fault: faulty / naive / NETRA", width="stretch")
    if (base / "gallery.png").exists():
        g2.image(str(base / "gallery.png"), caption="Faulty / repaired / ground truth", width="stretch")
    st.markdown('<div class="alert">Honest limits: blur is diagnosed but mostly not repairable; faults are simulated; '
                'no real-camera validation yet; no AWS deployment; the LLM planner was tested with mocks only.</div>',
                unsafe_allow_html=True)

st.markdown('<div class="foot">NETRA · OpenCV 5 · every action is logged and reversible</div>', unsafe_allow_html=True)
