# audio-tabula-rasa

**Can a generative model discover music from physics alone, with zero training on existing music?**

A research project exploring whether musical structure (consonance, intervals, eventually melody and rhythm) can emerge in a generator trained purely against a psychoacoustically-grounded reward model — no music corpus, no MIDI, no audio data of any kind.

> The deepest question: is music theory Turing-complete for musical taste? Or do you need experiential grounding in actual music for emotional resonance to emerge?

## The Hypothesis

Music theory is downstream of physics:
- Harmonic series comes from vibrating strings (Fourier analysis)
- Consonance/dissonance is a measurable basilar-membrane phenomenon (Plomp & Levelt 1965)
- Rhythm entrainment is neuroscience

If those underlying principles are enough, we should be able to bootstrap a music generator with **no copyrighted music in the loop** — by training a generator end-to-end against a reward model grounded only in psychoacoustic research and physics.

## Toy Case (this repo, phase 1)

The simplest possible version of the loop:

| Component | Implementation |
|---|---|
| Generator | 2-layer MLP → 2 frequencies (an interval) |
| Reward | Sethares (1993) dissonance over 6 harmonics |
| Training | REINFORCE policy gradient |
| Data used | **None** |

### Result

After ~1500 steps on a laptop CPU (under 2 minutes), the generator goes through a clean phase transition around step 750 and converges on the **major sixth / octave region** — the lowest-dissonance neighborhood under the Sethares model with 6 harmonics.

![Training summary](results/training_summary.png)

The right panel is the punchline: the reward landscape, derived from *physics alone*, has clear local minima at the canonical consonant intervals (3:2 perfect fifth, 2:1 octave). The generator finds them by climbing this landscape with no priors about music.

## Quick start

```bash
git clone https://github.com/ericspencer00/audio-tabula-rasa.git
cd audio-tabula-rasa
pip install -r requirements.txt
python -m src.train.reinforce --steps 1500
python -m src.train.plot
```

Or open `notebooks/01_toy_consonance.ipynb` in Colab — self-contained, runs in 2 minutes on CPU.

## Roadmap

- [x] **Phase 1 — Toy consonance.** Generator outputs 2 frequencies, reward = Sethares dissonance. Verify consonant intervals emerge. *(done — this repo)*
- [ ] **Phase 2 — Triads & voice leading.** 3-note chords. Reward extended with voice-leading smoothness penalty.
- [ ] **Phase 3 — Short melodies.** Sequence of N notes. Add temporal reward: scale coherence (via auto-correlation of pitch class set), contour smoothness.
- [ ] **Phase 4 — Rhythm.** Generator outputs onset times. Reward via rhythmic entrainment models (Large & Kolen 1994 oscillator networks).
- [ ] **Phase 5 — Spectrogram diffusion model.** Replace toy generator with a real audio model (diffusion over mel-spectrograms). Decode to waveform with a vocoder that was *not* trained on music (challenging — Griffin-Lim or learned-from-noise variants).
- [ ] **Phase 6 — RLAIF with a "taste" model.** Train a text-grounded music-theory taste model (read music theory textbooks, never hear music) and use it as the reward model in place of pure psychoacoustics.

## Hardware notes

- Toy case (this phase): runs in 2 minutes on any laptop CPU. Tested on M1 64GB and Linux.
- Phase 4 onward: will need GPU. Targeting 2× RTX 8000 (98GB VRAM) for spectrogram diffusion.

## References

- Plomp, R. & Levelt, W. J. M. (1965). *Tonal consonance and critical bandwidth.* JASA 38, 548–560.
- Sethares, W. A. (1993). *Local consonance and the relationship between timbre and scale.* JASA 94, 1218–1228.
- Large, E. W. & Kolen, J. F. (1994). *Resonance and the perception of musical meter.* Connection Science 6, 177–208.
- Silver, D. et al. (2017). *Mastering the game of Go without human knowledge.* Nature 550, 354–359. *(AlphaZero — the spiritual ancestor of this approach)*
- Bai, Y. et al. (2022). *Constitutional AI: Harmlessness from AI Feedback.* arXiv:2212.08073.

## License

MIT
