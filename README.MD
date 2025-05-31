Recursive Collapse Geometry (RCG)

A Universal Detection System for Flat Loop Collapse in Real-World SignalsCreated by Carney Pryately | 2025

Summary

Recursive Collapse Geometry (RCG) is the first open-source framework that defines and detects flat loop collapse — a universal structural signature observed in signals across domains like solar flares, seismic activity, and human voice.

RCG is the core detection layer of the broader Temporaxis System (not disclosed), offering a reproducible, signal-agnostic method for identifying recursive convergence in energy systems.

What is Flat Loop Collapse?

Flat Loop Collapse is a specific condition in a time-based signal where:

The rate of change becomes very low:|dΨ/dt| < δ₁

While the local energy remains high:∫Ψ²(x, y, t) dt > δ₂

This creates a recursive attractor — a brief window of stillness where energy recursively folds inward rather than flaring outward.

Core Features

Ψ(x, y, t) scaffold evolution field

Flat loop detection via dΨ/dt and energy threshold

Tested on real-world signals:

✅ Human speech (emotion bursts)

✅ Solar flare radio bursts (Cassini RPWS)

✅ Earthquake tremors

❌ White noise (no false positives)

Modular and compatible with .wav, .csv, or live input

📂 Repo Structure

RCG/
├── README.md
├── LICENSE
├── CITATION.cff
├── notebooks/
│   └── demo.ipynb
├── data/
│   ├── human.wav
│   ├── solar_flare_1.wav
│   ├── solar_flare_2.wav
│   ├── seismic_tremor.wav
│   └── white_noise.wav
├── plots/
│   ├── human_final_collapse.png
│   ├── solar_flare_1_final_collapse.png
│   ├── solar_flare_2_final_collapse.png
│   ├── seismic_tremor_final_collapse.png
│   └── white_noise_final_collapse.png
└── tscs/
    └── grid_flatloop.py

Claim of Origin

Recursive Collapse Geometry (RCG) and Flat Loop Collapse were first defined, implemented, and publicly released by Carney Pryately in 2025. This system includes the world’s first operational collapse detection scaffold Ψ(x, y, t), with threshold-based loop detection validated across solar, seismic, and cognitive domains. Any derivative use must cite this origin.

For Temporaxis-related licensing or collaboration inquiries: [neurolighthouse@gmail.com]

@misc{pryately2025rcg,
  title={Recursive Collapse Geometry},
  author={Carney Shea Pryately},
  year={2025},
  doi={10.5281/zenodo.15558528},
  howpublished={\url{https://github.com/carneyshea/Recursive-Collapse-Geometry}}