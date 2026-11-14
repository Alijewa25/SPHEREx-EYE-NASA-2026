"""
SPHEREx-EYE | Data Science yoxlama skripti
Işlət:  python validate.py
spectral_classifier.py ilə eyni qovluqda olmalıdır. Yalnız numpy lazımdır.

DİQQƏT: 1-ci və 2-ci test sintetik (model) spektrlərlədir, yəni kodun öz
daxili ardıcıllığını yoxlayır. Real SPHEREx ölçməsi DEYİL. Münsiflərə belə de.
"""
import numpy as np

try:
    from .spectral_classifier import (classify_candidate, snr_peak, candidate_snr,
                                      detection_metrics, _demo_spectrum)
except ImportError:
    from spectral_classifier import (classify_candidate, snr_peak, candidate_snr,
                                     detection_metrics, _demo_spectrum)

KINDS = ["H2O", "CH4", "CO2", "CO", "silikat", "none"]


def test_spectral_accuracy(n=200):
    print("\n=== 1) Sinifləndirmə dəqiqliyi (sintetik spektr, küy səviyyəsinə görə) ===")
    print("küy      " + "  ".join(f"{k:>7s}" for k in KINDS) + "   orta")
    for noise in [0.005, 0.01, 0.02, 0.05]:
        accs = []
        for kind in KINDS:
            ok = 0
            for s in range(n):
                w, f, e = _demo_spectrum(kind, noise=noise, seed=s)
                ok += classify_candidate(w, f, e)["class"] == kind
            accs.append(ok / n)
        print(f"{noise:<8} " + "  ".join(f"{a:7.1%}" for a in accs) + f"  {np.mean(accs):6.1%}")


def test_false_alarm(n=1000):
    print("\n=== 2) Yalançı həyəcan (yalnız küy; 'none' olmayan cavab = səhv) ===")
    for noise in [0.01, 0.02, 0.05]:
        bad = 0
        for s in range(n):
            w, f, e = _demo_spectrum("none", noise=noise, seed=10_000 + s)
            bad += classify_candidate(w, f, e)["class"] != "none"
        print(f"küy {noise}: {bad / n:.1%}  ({bad}/{n})")


def _fake_image(amp, seed, sigma_bkg=5.0):
    rng = np.random.default_rng(seed)
    img = rng.normal(100, sigma_bkg, (100, 100))
    yy, xx = np.indices(img.shape)
    img += amp * np.exp(-((xx - 50) ** 2 + (yy - 50) ** 2) / (2 * 1.5 ** 2))
    return img


def test_snr(n=50):
    print("\n=== 3) SNR yoxlaması (parlaqlığı bilinən süni ulduz) ===")
    print("amplituda  gözlənən SNR   snr_peak(orta)   aperture SNR(orta)")
    for amp in [10, 25, 50, 100, 300]:
        pk = [snr_peak(_fake_image(amp, s), 50, 50) for s in range(n)]
        ap = [candidate_snr(_fake_image(amp, s), 50, 50)["snr"] for s in range(n)]
        print(f"{amp:<10} {amp / 5:<14.1f} {np.mean(pk):<16.1f} {np.mean(ap):.1f}")
    print("(snr_peak gözlənənə yaxın olmalıdır; kiçik amplitudada küy üzündən bir az yuxarı çıxa bilər)")


# BA-nın verdiyi hədəflər (PM dəyərləri SIMBAD-da bir dəfə yoxlanmalıdır)
#  (ad, RA, DEC, PM "/il, DSS-də görünür?)  - WISE 0855 və Luhman 16 yalnız IR hekayəsi üçündür
TARGETS = [
    ("Barnard",    269.4521,  4.6933, 10.3, True),
    ("Kapteyn",     77.9192, -45.0183, 8.6, True),
    ("Proxima Cen", 217.4288, -62.6794, 3.8, True),
    ("Luhman 16",  162.3280, -53.3194, 2.8, False),
    ("WISE 0855",  133.7951,  -7.2451, 8.1, False),
]


def test_motion(years=43.0, fov_arcmin=15.0, px=600):
    scale = fov_arcmin * 60 / px
    print(f"\n=== 4) Gözlənən hərəkət ({years:.0f} il, FOV {fov_arcmin}', {scale:.2f}\"/piksel) ===")
    print("hədəf         PM(\"/il)  hərəkət(\")  hərəkət(piksel)  kadra sığır?")
    for name, ra, dec, pm, dss in TARGETS:
        mot = pm * years
        fits = "bəli" if mot / scale < px * 0.9 else "XEYR - FOV artır"
        note = "" if dss else "  [DSS demosuna qoyma, yalnız IR hekayəsi]"
        print(f"{name:<13} {pm:<9} {mot:<11.0f} {mot / scale:<16.0f} {fits}{note}")
    print("Backend-in ölçdüyü motion_arcsec bu rəqəmlə uyğun gəlməlidir (±10%).")
    print("Real tarixləri (years) FITS DATE-OBS-dan hesabla, 43 yazma.")


def report(rows):
    """rows: [(sahə_adı, tapılan_xy, həqiqi_xy, snr_siyahısı), ...] -> cədvəl"""
    print("\nsahə          precision  recall    F1     TP FP FN  orta SNR")
    P, R = [], []
    for name, det, tru, snrs in rows:
        m = detection_metrics(det, tru)
        P.append(m["precision"]); R.append(m["recall"])
        s = np.mean(snrs) if len(snrs) else float("nan")
        print(f"{name:<13} {m['precision']:<10} {m['recall']:<9} {m['f1']:<6} "
              f"{m['TP']:<2} {m['FP']:<2} {m['FN']:<3} {s:.1f}")
    print(f"ORTA          {np.mean(P):.3f}      {np.mean(R):.3f}")


def demo_metrics():
    print("\n=== 5) Metrik cədvəli: NÜMUNƏ (süni rəqəmlər, REAL NƏTİCƏ DEYİL) ===")
    rng = np.random.default_rng(0)
    rows = []
    for name in ["sahə1", "sahə2", "sahə3", "sahə4", "sahə5"]:
        tru = [tuple(rng.uniform(50, 550, 2)) for _ in range(5)]
        det = [(x + rng.normal(0, 1), y + rng.normal(0, 1)) for x, y in tru[:4]]
        det += [tuple(rng.uniform(50, 550, 2)) for _ in range(rng.integers(0, 3))]
        rows.append((name, det, tru, rng.uniform(8, 30, len(det))))
    report(rows)
    print("\nReal nəticə üçün: backend-in tapdığı centroid-ləri və SIMBAD həqiqi\n"
          "koordinatlarını (piksellə) report([...]) funksiyasına ver.")


if __name__ == "__main__":
    test_spectral_accuracy()
    test_false_alarm()
    test_snr()
    test_motion()
    demo_metrics()
