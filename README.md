# SPHEREx-EYE: AI Deep-Sky Planet & Multi-Spectral Anomaly Hunter

**NASA Space Apps Challenge 2026 — Baku Local Event**  
**Team:** HERschel (`T06`)  
**Challenge:** Planet X and SPHEREx  

---

## Overview
NASA's SPHEREx mission maps the entire sky every 6 months across **102 infrared spectral bands (0.75 – 5.0 µm)**, generating massive multi-epoch time-series datasets. Manually inspecting billions of celestial sources to detect faint, slow-moving bodies—such as the hypothetical **Planet X (Planet 9)**, Kuiper Belt Objects (KBOs), brown dwarfs, and Near-Earth Asteroids (NEOs)—is computationally prohibitive with traditional pipelines.

**SPHEREx-EYE** is an AI-powered Computer Vision and Spectral Analysis platform that aligns multi-epoch sky maps, isolates transient/moving anomalies via sub-pixel image differencing, and classifies candidates using their 102-channel thermal Spectral Energy Distribution (SED).

---

## Key Features
- **Sub-Pixel Multi-Epoch Alignment & Differencing:** Eliminates static background stars and galaxies between Epoch 1 ($T_0$) and Epoch 2 ($T_0 + 6\text{ months}$) to isolate moving objects.
- **102-Band Infrared Spectral Fingerprinting:** Analyzes thermal radiation curves and molecular absorption lines (methane at $3.32\,\mu\text{m}$, silicates at $2.95\,\mu\text{m}$) to distinguish cold outer-solar-system planets (~48 K) from warmer asteroids (~274 K) and brown dwarfs.
- **Interactive Blink Comparator:** Web-based astronomical inspection tool allowing researchers and citizen scientists to visually verify orbital shifts.
- **NASA FITS Native Support:** Directly ingests astronomical `.fits` / `.fit` files via `Astropy` as well as standard optical/IR imagery.

---

## Project Architecture
```text
SPHEREx-EYE-NASA-2026/
├── backend/
│   ├── app/
│   │   ├── ai/
│   │   │   └── detector.py      # CV alignment, differencing & 102-band spectral classifier
│   │   └── main.py              # FastAPI REST endpoints
│   └── requirements.txt         # Python dependencies
├── frontend/
│   └── index.html               # Interactive Blink Comparator & Chart.js Spectral UI
└── README.md
```
##Quick Start (Local Setup)
Clone the repository:

```bash
git clone [https://github.com/Alijewa25/SPHEREx-EYE-NASA-2026.git](https://github.com/Alijewa25/SPHEREx-EYE-NASA-2026.git)
cd SPHEREx-EYE-NASA-2026
```

##Install Python dependencies:

```Bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

Run the FastAPI backend:

```Bash
uvicorn backend.app.main:app --reload
```
##Launch the Frontend:
```
Open frontend/index.html in your browser (or via VS Code Live Server).
```

#Team HERschel
* Full-Stack & AI Pipeline Architecture
* Data Science & Astrophysical Modeling
* UX/UI Design & Interactive Visualization
* Business & Scientific Impact Analysis
* Pitch & Mission Presentation
