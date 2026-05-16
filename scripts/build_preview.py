"""
Build the GitHub Pages audio-preview site.

Reads the rendered WAVs in `results/audio/` and the per-phase plots
in `results/phase*/`, then writes a single static `preview/index.html`
that plays every phase's audio in the browser and shows the matching
training-summary plot beside it.

Run this after `python -m src.render.render_phases`.
"""
import html
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / "preview"
AUDIO_DIR = ROOT / "results" / "audio"


# One entry per playable phase. order is the page layout.
PHASES = [
    {
        "id": "song",
        "name": "Phase 14 — First composed song (16 bars)",
        "wav": "song_first.wav",
        "blurb": (
            "Multi-track arrangement of trained generators: Phase-2 "
            "progressions on a pad, plucked bass on the chord roots, "
            "Phase-3 melodies quantized to a major pentatonic on the "
            "lead, and a 4/4 backbeat. Closer to listenable music than "
            "the per-phase demos but still robotic — work in progress."
        ),
    },
    {
        "id": "phase1",
        "name": "Phase 1 — Interval discovery",
        "wav": "phase1_intervals.wav",
        "plot": "results/training_summary.png",
        "blurb": (
            "Two-frequency generator + Sethares dissonance reward. "
            "Consonant intervals (major sixth / octave region) emerge "
            "with no music data."
        ),
    },
    {
        "id": "phase2_triads",
        "name": "Phase 2 — Triads",
        "wav": "phase2_triads.wav",
        "plot": "results/phase2_triads/triad_summary.png",
        "blurb": (
            "3-voice generator + pairwise Sethares + voice spread. "
            "Discovers sus4 (6:8:9), major (4:5:6), augmented, and "
            "diminished chords; prefers the upper register."
        ),
    },
    {
        "id": "phase2_prog",
        "name": "Phase 2 — Chord progressions",
        "wav": "phase2_progressions.wav",
        "plot": "results/phase2_progressions/progression_summary.png",
        "blurb": (
            "Adds Tymoczko-style voice-leading cost. Mixed canonical "
            "triads with smooth voice movement."
        ),
    },
    {
        "id": "phase3",
        "name": "Phase 3 — Monophonic melodies",
        "wav": "phase3_melodies.wav",
        "plot": "results/phase3_melodies/melody_summary.png",
        "blurb": (
            "Sequential Sethares + Terhardt virtual-pitch salience + "
            "pitch-class diversity. 4–5-PC melodic gestures emerge — "
            "the top Western-scale match is the blues scale."
        ),
    },
    {
        "id": "phase4",
        "name": "Phase 4 — Rhythm",
        "wav": "phase4_rhythms.wav",
        "plot": "results/phase4_rhythms/rhythm_summary.png",
        "blurb": (
            "Phase-coherence-based entrainment reward (linear "
            "approximation to Large–Kolen 1994). Discovered tempo "
            "peaks at ~120 BPM — inside Fraisse's preferred-tempo window."
        ),
    },
    {
        "id": "phase34",
        "name": "Phase 3+4 — Cross-paired melodic rhythm",
        "wav": "phase34_melodic_rhythm.wav",
        "blurb": (
            "Phase-3 melody pitches placed at Phase-4 rhythm onsets — "
            "no joint training, just synthesis composition."
        ),
    },
    {
        "id": "phase45",
        "name": "Phase 4.5 — Joint melodic rhythm",
        "wav": "phase45_melodic_rhythm.wav",
        "plot": "results/phase4_5_melodic_rhythm/melodic_rhythm_summary.png",
        "blurb": (
            "Single MLP emits (pitch, IOI) pairs. Reward = melody + "
            "rhythm, jointly optimized. Tonal salience reaches 0.73 "
            "while phase coherence holds at 0.69."
        ),
    },
    {
        "id": "phase7",
        "name": "Phase 7 — 2-voice counterpoint",
        "wav": "phase7_counterpoint.wav",
        "plot": "results/phase7_counterpoint/counterpoint_summary.png",
        "blurb": (
            "Banded per-voice generator + horizontal/vertical Sethares + "
            "voice-crossing penalty. Zero crossings, P5–octave vertical "
            "intervals."
        ),
    },
    {
        "id": "phase13_3v",
        "name": "Phase 13 — 3-voice chorale",
        "wav": "phase13_3voice_counterpoint.wav",
        "plot": "results/phase13_3voice_counterpoint/counterpoint_summary.png",
        "blurb": (
            "Same architecture, n_voices=3. Best-checkpoint reward +5.33, "
            "stratified voice lines with zero crossings."
        ),
    },
    {
        "id": "phase13_4v",
        "name": "Phase 13 — 4-voice chorale",
        "wav": "phase13_4voice_counterpoint.wav",
        "plot": "results/phase13_4voice_counterpoint/counterpoint_summary.png",
        "blurb": (
            "n_voices=4. Six vertical pairs to satisfy — the limit of "
            "banded-MLP + REINFORCE at this training budget."
        ),
    },
    {
        "id": "phase8b",
        "name": "Phase 8b — Bohlen-Pierce triads (odd-partial timbre)",
        "wav": "phase8b_bp_triads_odd_timbre.wav",
        "plot": "results/phase8b_bp_triads/bp_triads_summary.png",
        "blurb": (
            "Same triad generator, partials=odd. Discovered chords "
            "cluster on BP-style ratios (≈ 5:7:9). Rendered with "
            "odd-only-harmonic synthesis so you hear the matching timbre."
        ),
    },
]


def _git_short_sha():
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT
        ).decode().strip()
    except Exception:
        sha = os.environ.get("GITHUB_SHA", "local")[:7]
    return sha


def _git_subject():
    try:
        return subprocess.check_output(
            ["git", "log", "-1", "--pretty=%s"], cwd=ROOT
        ).decode().strip()
    except Exception:
        return ""


def main():
    PREVIEW.mkdir(parents=True, exist_ok=True)
    audio_out = PREVIEW / "audio"
    audio_out.mkdir(exist_ok=True)
    plot_out = PREVIEW / "plots"
    plot_out.mkdir(exist_ok=True)

    sections = []
    for phase in PHASES:
        wav_src = AUDIO_DIR / phase["wav"]
        if not wav_src.exists():
            print(f"[skip] missing {wav_src}", file=sys.stderr)
            continue
        shutil.copy2(wav_src, audio_out / wav_src.name)

        plot_html = ""
        if "plot" in phase:
            plot_src = ROOT / phase["plot"]
            if plot_src.exists():
                shutil.copy2(plot_src, plot_out / f"{phase['id']}.png")
                plot_html = (
                    f'<img class="plot" src="plots/{phase["id"]}.png" '
                    f'alt="{html.escape(phase["name"])} plot">'
                )

        sections.append(
            dedent(f"""
            <section id="{phase['id']}">
              <h2>{html.escape(phase['name'])}</h2>
              <p>{html.escape(phase['blurb'])}</p>
              <audio controls preload="metadata"
                     src="audio/{phase['wav']}"></audio>
              {plot_html}
            </section>
        """).strip()
        )

    sha = _git_short_sha()
    subject = _git_subject()
    body = "\n".join(sections)

    report = ""
    rpt_path = ROOT / "results" / "QUANTITATIVE_REPORT.txt"
    if rpt_path.exists():
        report = html.escape(rpt_path.read_text())

    html_doc = dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <title>Audio Tabula Rasa — preview</title>
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <style>
            :root {{ color-scheme: light dark; }}
            body {{ font-family: ui-sans-serif, system-ui, sans-serif;
                   max-width: 860px; margin: 2rem auto; padding: 0 1rem;
                   line-height: 1.55; }}
            h1 {{ margin-bottom: 0.2rem; }}
            .commit {{ color: #888; font-family: ui-monospace, monospace;
                       margin-top: 0; font-size: 0.92rem; }}
            section {{ margin: 2.4rem 0; padding-top: 1.6rem;
                       border-top: 1px solid #ccc4; }}
            section:first-of-type {{ border-top: 0; padding-top: 0; }}
            h2 {{ margin-bottom: 0.4rem; }}
            audio {{ width: 100%; margin: 0.4rem 0 0.8rem; }}
            .plot {{ width: 100%; height: auto; border-radius: 6px;
                     box-shadow: 0 1px 3px #0002; }}
            pre.report {{ white-space: pre-wrap; font-size: 0.78rem;
                          background: #0001; padding: 1rem; border-radius: 6px;
                          overflow-x: auto; }}
          </style>
        </head>
        <body>
          <h1>audio-tabula-rasa</h1>
          <p class="commit">commit <a href="https://github.com/EricSpencer00/audio-tabula-rasa/commit/{sha}">{sha}</a> — {html.escape(subject)}</p>
          <p>
            A music generator trained against psychoacoustic physics rewards
            only — no MIDI, no audio corpus, no learned vocoder. Every audio
            file below is freshly rendered from the model weights in this
            commit by an additive sine-bank synthesizer.
          </p>
          {body}
          <section id="report">
            <h2>Quantitative report</h2>
            <p>Full statistics across all phases:</p>
            <pre class="report">{report}</pre>
          </section>
          <p class="commit" style="margin-top:3rem;">
            built by .github/workflows/audio-preview.yml
          </p>
        </body>
        </html>
    """).strip()

    (PREVIEW / "index.html").write_text(html_doc)
    print(f"wrote {PREVIEW/'index.html'}")
    print(f"copied {len(list(audio_out.iterdir()))} WAVs to {audio_out}")
    print(f"copied {len(list(plot_out.iterdir()))} plots to {plot_out}")


if __name__ == "__main__":
    main()
