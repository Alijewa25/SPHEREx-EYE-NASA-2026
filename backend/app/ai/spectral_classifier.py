import numpy as np





_SPHEREX_BANDS = [
    (0.75, 1.11, 41), (1.11, 1.64, 41), (1.64, 2.42, 41),
    (2.42, 3.82, 35), (3.82, 4.42, 110), (4.42, 5.00, 130),
]


def spherex_wavelengths():

    centers = []
    for lo, hi, _ in _SPHEREX_BANDS:
        edges = np.geomspace(lo, hi, 18)
        centers.append(np.sqrt(edges[:-1] * edges[1:]))
    return np.concatenate(centers)






FEATURES = {
    "H2O_1.5":    ("H2O",      (1.44, 1.62), (1.25, 1.35), (1.80, 1.88)),
    "H2O_2.0":    ("H2O",      (1.95, 2.10), (1.80, 1.88), (2.15, 2.22)),
    "H2O_3.0":    ("H2O",      (2.85, 3.20), (2.50, 2.65), (3.60, 3.75)),
    "CH4_1.72":   ("CH4",      (1.69, 1.75), (1.64, 1.67), (1.77, 1.79)),
    "CH4_2.32":   ("CH4",      (2.28, 2.38), (2.20, 2.25), (2.45, 2.50)),
    "CH4_3.32":   ("CH4",      (3.28, 3.38), (3.20, 3.24), (3.45, 3.55)),


    "SIL_1.0":    ("silikat",  (0.90, 1.05), (0.75, 0.82), (1.20, 1.30)),
    "CO2_4.27":   ("CO2",      (4.22, 4.32), (4.05, 4.15), (4.40, 4.50)),
    "CO_4.67":    ("CO",       (4.64, 4.70), (4.52, 4.58), (4.80, 4.90)),
}

LABELS = {
    "H2O": "Su buzu zəngin (kometa / buzlu TNO tipi)",
    "CH4": "Metan buzu (Pluton tipli uzaq cisim)",
    "silikat": "Silikat (yalnız zəif 1 µm işarəsi; əsas 9.7 µm zolağı aralıqdan kənardadır)",
    "CO2": "CO2 buzu (4.27 µm)",
    "CO": "CO buzu (4.67 µm)",
    "none": "Spektral zolaq tapılmadı (ulduz / qaya olmayan kontinuum)",
}


def _estimate_err(flux):

    d = np.diff(flux)
    return np.full_like(flux, 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2) + 1e-12)


def band_depth(wl, flux, err, band, left, right):

    mb = (wl >= band[0]) & (wl <= band[1])
    ml = (wl >= left[0]) & (wl <= left[1])
    mr = (wl >= right[0]) & (wl <= right[1])
    if mb.sum() < 1 or ml.sum() < 1 or mr.sum() < 1:
        return None

    xl, yl = wl[ml].mean(), flux[ml].mean()
    xr, yr = wl[mr].mean(), flux[mr].mean()
    cont = yl + (yr - yl) * (wl[mb] - xl) / (xr - xl)
    if np.any(cont <= 0):
        return None

    depth = 1.0 - np.mean(flux[mb] / cont)
    xc = wl[mb].mean()
    wr = (xc - xl) / (xr - xl)
    wl_ = 1 - wr
    s_b2 = np.sum(err[mb] ** 2) / mb.sum() ** 2
    s_l2 = np.mean(err[ml] ** 2) / ml.sum()
    s_r2 = np.mean(err[mr] ** 2) / mr.sum()
    sigma = np.sqrt(s_b2 + wl_ ** 2 * s_l2 + wr ** 2 * s_r2) / np.mean(cont)
    return depth, depth / sigma


def classify_candidate(wavelengths_um, flux, flux_err=None,
                       min_sigma=3.0, min_depth=0.03):

    wl = np.asarray(wavelengths_um, float)
    fl = np.asarray(flux, float)
    err = _estimate_err(fl) if flux_err is None else np.asarray(flux_err, float)

    order = np.argsort(wl)
    wl, fl, err = wl[order], fl[order], err[order]

    bands = {}
    scores = {"H2O": 0.0, "CH4": 0.0, "CO2": 0.0, "CO": 0.0, "silikat": 0.0}
    for name, (mat, band, left, right) in FEATURES.items():
        res = band_depth(wl, fl, err, band, left, right)
        if res is None:
            continue
        depth, sig = res
        found = bool(depth >= min_depth and sig >= min_sigma)
        bands[name] = {"depth": round(float(depth), 4),
                       "sigma": round(float(sig), 2), "detected": found}
        if found:
            scores[mat] += sig


    if scores["H2O"] > scores["silikat"]:
        scores["silikat"] *= 0.5

    total = sum(scores.values())
    if total == 0:
        cls, conf = "none", 0.0
    else:
        cls = max(scores, key=scores.get)
        conf = scores[cls] / total


    lo = fl[(wl > 0.75) & (wl < 1.2)].mean()
    hi = fl[(wl > 2.0) & (wl < 2.6)].mean()
    slope = float(np.log(hi / lo)) if lo > 0 and hi > 0 else float("nan")

    return {"class": cls, "label": LABELS[cls], "confidence": round(float(conf), 3),
            "spectral_source": "model",
            "scores": {k: round(float(v), 2) for k, v in scores.items()},
            "bands": bands, "slope": round(slope, 3)}





def candidate_snr(image, x, y, r_ap=4, r_in=8, r_out=12):

    img = np.asarray(image, float)
    yy, xx = np.indices(img.shape)
    r = np.hypot(xx - x, yy - y)
    ap = r <= r_ap
    ann = (r >= r_in) & (r <= r_out)
    ann_vals = img[ann & np.isfinite(img)]
    if ap.sum() == 0 or ann_vals.size < 10:
        return {"snr": float("nan")}


    med, std = np.median(ann_vals), ann_vals.std()
    ann_vals = ann_vals[np.abs(ann_vals - med) < 3 * std]
    bkg, noise = ann_vals.mean(), ann_vals.std()

    n_ap = ap.sum()
    signal = np.nansum(img[ap]) - n_ap * bkg
    snr = signal / (noise * np.sqrt(n_ap) + 1e-12)
    return {"snr": float(snr), "signal": float(signal),
            "background": float(bkg), "noise": float(noise)}


def snr_peak(image, x, y, r_in=8, r_out=12, r_peak=2):

    img = np.asarray(image, float)
    margin = r_out + 1
    x0, x1 = max(0, int(x - margin)), min(img.shape[1], int(x + margin + 1))
    y0, y1 = max(0, int(y - margin)), min(img.shape[0], int(y + margin + 1))
    local = img[y0:y1, x0:x1]
    yy, xx = np.indices(local.shape)
    r = np.hypot(xx + x0 - x, yy + y0 - y)
    finite = np.isfinite(local)
    ann_vals = local[(r >= r_in) & (r <= r_out) & finite]
    core = local[(r <= r_peak) & finite]
    if ann_vals.size < 10 or core.size == 0:
        return float("nan")
    med, std = np.median(ann_vals), ann_vals.std()
    ann_vals = ann_vals[np.abs(ann_vals - med) < 3 * std]
    return float((core.max() - np.median(ann_vals)) / (ann_vals.std() + 1e-12))


def detection_metrics(detected_xy, truth_xy, match_radius=5.0):

    det = [tuple(p) for p in detected_xy]
    tru = [tuple(p) for p in truth_xy]
    used, tp = set(), 0
    for d in det:
        best, best_d = None, match_radius
        for i, t in enumerate(tru):
            if i in used:
                continue
            dist = np.hypot(d[0] - t[0], d[1] - t[1])
            if dist <= best_d:
                best, best_d = i, dist
        if best is not None:
            used.add(best)
            tp += 1
    fp, fn = len(det) - tp, len(tru) - tp
    prec = tp / (tp + fp) if det else 0.0
    rec = tp / (tp + fn) if tru else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(f1, 3), "TP": tp, "FP": fp, "FN": fn}





def _demo_spectrum(kind, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    wl = spherex_wavelengths()
    f = 1.0 + 0.15 * (wl - 0.75)

    def dip(c, w, d):
        return d * np.exp(-0.5 * ((wl - c) / w) ** 2)

    if kind == "H2O":
        f = f * (1 - dip(1.5, 0.05, 0.25) - dip(2.0, 0.06, 0.3) - dip(3.0, 0.2, 0.6))
    elif kind == "CH4":
        f = f * (1 - dip(1.72, 0.02, 0.15) - dip(2.32, 0.03, 0.25) - dip(3.32, 0.04, 0.3))
    elif kind == "silikat":
        f = f * (1 - dip(1.0, 0.07, 0.2))
    elif kind == "CO2":
        f = f * (1 - dip(4.27, 0.04, 0.4))
    elif kind == "CO":
        f = f * (1 - dip(4.67, 0.03, 0.3))
    return wl, f + rng.normal(0, noise, wl.size), np.full(wl.size, noise)


if __name__ == "__main__":
    for k in ["H2O", "CH4", "CO2", "CO", "silikat", "none"]:
        w, f, e = _demo_spectrum(k)
        r = classify_candidate(w, f, e)
        print(f"{k:8s} -> {r['class']:8s} etibar={r['confidence']}  {r['scores']}")
