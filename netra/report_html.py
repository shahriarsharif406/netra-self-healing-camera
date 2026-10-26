"""One-file HTML maintenance report for a repair run: shareable with a maintenance team, prints cleanly."""
from __future__ import annotations
import base64
import datetime as dt
import html
import cv2
import numpy as np


def _img(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _e(x) -> str:
    return html.escape(str(x))


def build_report(res, receipt: dict, chain_ok: bool, n_events: int, forecasts: list[dict] | None = None,
                 camera: str = "camera") -> str:
    steps = "".join(
        f"<tr><td>{_e(s['fault'])}</td><td><code>{_e(s['tool'])}</code></td><td class='{_e(s['status'])}'>{_e(s['status'])}</td>"
        f"<td>{_e(s['decision'])}</td><td>{_e(s['note'])}</td></tr>" for s in res.steps) or "<tr><td colspan=5>No action taken.</td></tr>"
    diag = "".join(f"<tr><td>{_e(f.fault)}</td><td>{f.severity:.2f}</td></tr>" for f in res.initial.findings) \
        or "<tr><td colspan=2>No fault found.</td></tr>"
    alerts = "".join(f"<li>{_e(a)}</li>" for a in res.alerts) or "<li>None.</li>"
    fc = "".join(f"<li>{_e(f['message'])}</li>" for f in (forecasts or [])) or "<li>No forecast attached.</li>"
    unresolved = ", ".join(res.unresolved) or "none"
    chain = (f"Audit chain verified ({n_events} hash-linked events)." if chain_ok
             else "WARNING: audit chain could not be verified.")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NETRA camera health report - {_e(camera)}</title>
<style>
body{{font:15px/1.55 Inter,system-ui,sans-serif;background:#f6f8fc;color:#0f172a;margin:0}}
main{{max-width:960px;margin:0 auto;padding:32px 20px}}
header{{background:linear-gradient(135deg,#0d1730,#2a1b57);color:#fff;border-radius:18px;padding:26px 30px}}
header small{{color:#67e8f9;letter-spacing:.16em;text-transform:uppercase;font:600 11px monospace}}
h1{{margin:4px 0 2px;font-size:30px}} h2{{margin:28px 0 8px;font-size:16px;text-transform:uppercase;letter-spacing:.08em;color:#475569}}
.card{{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:16px 18px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}} img{{width:100%;border-radius:10px;border:1px solid #e2e8f0}}
table{{width:100%;border-collapse:collapse;font-size:14px}} th,td{{text-align:left;padding:7px 8px;border-bottom:1px solid #e2e8f0;vertical-align:top}}
th{{color:#64748b;font-weight:600}} code{{background:#eef2ff;padding:1px 6px;border-radius:5px}}
.applied{{color:#059669;font-weight:600}}.partial{{color:#b45309;font-weight:600}}.rolled_back,.failed{{color:#dc2626;font-weight:600}}.denied{{color:#d97706;font-weight:600}}
.note{{background:#fffbeb;border:1px solid #fde68a;border-radius:12px;padding:10px 14px;color:#92400e}}
.hash{{font:12px monospace;word-break:break-all}} @media print{{body{{background:#fff}}}}
</style></head><body><main>
<header><small>NETRA &middot; camera health report</small><h1>{_e(camera)}</h1>
<div>Generated {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} &middot; run {_e(res.run_id)} &middot; {res.runtime_ms:.0f} ms</div></header>
<h2>Before / after</h2><div class="grid"><div class="card"><b>Before</b><img src="{_img(res.original)}"></div>
<div class="card"><b>After</b><img src="{_img(res.output)}"></div></div>
<h2>Diagnosis (severity 0-1)</h2><div class="card"><table><tr><th>Fault</th><th>Severity</th></tr>{diag}</table></div>
<h2>What the agent did</h2><div class="card"><table><tr><th>Fault</th><th>Tool</th><th>Status</th><th>Decision</th><th>Verification</th></tr>{steps}</table></div>
<h2>Needs a human</h2><div class="card">{_e(unresolved)}</div>
<h2>Maintenance alerts</h2><div class="card"><ul>{alerts}</ul></div>
<h2>Forecast (Health Passport)</h2><div class="card"><ul>{fc}</ul></div>
<h2>Provenance</h2><div class="card"><p>{_e(chain)}</p>
<p class="hash">Original frame SHA-256: {_e(receipt['original_sha256'])}<br>Repaired frame SHA-256: {_e(receipt['repaired_sha256'])}<br>Chain head: {_e(receipt['chain_head'])}</p></div>
<p class="note" style="margin-top:18px">{_e(receipt['notice'])} Integrity evidence, not a digital signature. Faults and numbers here come from the frames provided; see the project README for evaluation limits.</p>
</main></body></html>"""
