# Devpost submission kit

**Deadline: 27 Oct 2026, 12:45 pm (GMT+6).** Aim to submit by 9:00 am.

## Tagline
A camera agent that diagnoses why its own feed is degraded, repairs it with OpenCV 5, verifies the repair, and asks a human before risky changes.

## Inspiration
Vision systems rarely fail loudly. A smudged lens, a bumped mount or a dark evening silently degrades every model downstream. Most work improves the model; NETRA fixes the camera feed.

## What it does
Given a reference frame from the healthy camera, NETRA diagnoses six faults (blur, low light, overexposure, noise, camera drift, lens dirt), chooses a repair from a fixed tool registry, applies it, re-diagnoses to verify it helped, rolls it back if not, and escalates what it cannot fix. Medium-risk changes (re-alignment, regional edits) need human approval; with no approver they are denied. Every step goes to a JSONL audit log, and maintenance alerts name the likely physical cause.

## How we built it
Python and OpenCV 5.0: ORB/RANSAC and ECC alignment, Laplacian and Immerkaer measurements, CLAHE, bilateral and non-local-means filters, `cv2.dft` Wiener deconvolution, inpainting. A rule-based planner by default; an optional LLM planner can only propose actions that are validated against the diagnosis and tool registry. Streamlit UI, CLI, 24 tests.

## Results (fresh held-out scenes, 192 faulty frames)
Mean SSIM 0.498 -> 0.843, 0 frames made worse, 0/16 false alarms on clean frames; a naive "always enhance" pipeline made 71% of frames worse. About 650 ms per frame on 2 CPU cores, no GPU.

**Health Passport:** records every diagnosis per camera and forecasts when a fault limit will be crossed (Theil-Sen + Mann-Kendall). On simulated slow degradations at day 30: drift 6/6 (1.1 days error), noise 6/6 (1.1), focus loss 6/6 (2.3), dimming 6/6 (4.9); 0 false alerts on stable cameras. Early dirt forecasts are poor (~16 days error).
**Task guard:** rolls back repairs that lower the user's downstream score: QR decoding in fixed regions 0.72 -> 0.79, and 0 frames where the task got worse than the input (17 without the guard), at the cost of SSIM 0.870 -> 0.829.

## Challenges
Our evaluation exposed a flaw: after exposure repairs, clipped regions looked like lens smudges (73 false alerts). A first fix made it worse and was discarded; excluding pixels the sensor clipped fixed it (calibration 5/20 -> 0/20 false alarms). We re-ran on fresh scenes and report only those.

## Limitations (say these in the video too)
Blur is diagnosed but mostly not repairable. Evaluation uses simulated faults and a fixed camera with a reference frame. No real-camera validation, no AWS deployment, no COOL integration, LLM planner tested with mocks only.

## What's next
Real-camera footage, real long-term camera footage for the Passport, learned deblurring behind the same verifier, AWS Graviton deployment.

## Built with
python, opencv, numpy, streamlit, scikit-image

## 3-minute video script
* 0:00 Problem: "Your vision system is only as good as your camera. A lens smudge, a bumped mount, a dark evening, and nothing alerts you." Show gallery.png.
* 0:25 Live demo (UI): inject drift + noise, show the diagnosis table, press Repair with geometry approval OFF: drift is **denied**, noise is fixed automatically. Tick the approval, repair again: SSIM jumps.
* 1:15 Show the audit log and the maintenance alerts.
* 1:35 Show a rolled-back step (blur) and explain verification.
* 1:50 Passport tab: pick 'creeping drift', day 20, show the forecast warning. Then the task guard result.
* 2:00 Results slide: SSIM bars, "naive makes 71% worse, NETRA 0%".
* 2:25 Honest slide: blur not solved, simulated faults, how the dirt false alarms were found and fixed.
* 2:45 Close: architecture diagram, repo link.

## Checklist
- [ ] Push the repo to GitHub (public) and add the link
- [ ] Record the video (3 min max), upload to YouTube or Vimeo, add the link
- [ ] Paste the sections above into Devpost
- [ ] Upload gallery.png and fig_ssim_by_fault.png as images
- [ ] Do NOT claim AWS deployment or COOL unless you actually do it
- [ ] Optional: open an AWS account and run deploy/aws_deploy.sh, then update README "Not done"
