"""
SPHEREx-EYE | Data Science <-> backend birləşdirmə modulu

Backend (detector.py) hər namizəd üçün bu modulun analyze_candidate() funksiyasını
çağırır, nəticələri CSV-yə yazır. Üç fayl eyni qovluqda olmalıdır:
    spectral_classifier.py, validate.py, ds_integration.py
Yalnız numpy lazımdır (WCS istifadə olunarsa astropy də).
"""
import csv
import re
import datetime
import numpy as np

try:
    from .spectral_classifier import classify_candidate, snr_peak
except ImportError:
    from spectral_classifier import classify_candidate, snr_peak

# BA-nın tələb etdiyi CSV sütunları, eyni sıra ilə
CSV_COLUMNS = [
    "candidate_id", "ra_deg", "dec_deg", "survey_epoch1", "survey_epoch2",
    "date_obs1", "date_obs2", "pixel_scale_arcsec", "centroid1_x", "centroid1_y",
    "centroid2_x", "centroid2_y", "motion_arcsec", "pm_arcsec_yr", "snr",
    "spectral_class", "spectral_source", "confidence", "data_source",
    "known_match", "review_status",
]


# ---------- FITS başlığından tarix və piksel miqyası ----------
def parse_date_obs(value):
    """ISO ('1953-09-23...') və köhnə DSS ('23/09/53') formatlarını oxuyur."""
    if value is None:
        return None
    s = str(value).strip()
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    try:
        if m:
            return datetime.date(int(m[1]), int(m[2]), int(m[3]))
        m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", s)
        if m:
            y = int(m[3])
            y = 1900 + y if y < 100 else y
            return datetime.date(y, int(m[2]), int(m[1]))
    except ValueError:
        pass
    return None


def pixel_scale_arcsec(header):
    """Piksel miqyası (qövs saniyəsi/piksel). Tapılmasa None."""
    for key in ("CDELT2", "CDELT1"):
        v = header.get(key)
        if v:
            return abs(float(v)) * 3600.0
    c11, c21 = header.get("CD1_1"), header.get("CD2_1")
    if c11 is not None and c21 is not None:
        return float(np.hypot(c11, c21)) * 3600.0
    return None


# ---------- Saxta müsbət filtrləri ----------
def near_bright_star(image, x, y, radius=20, core=3, bright_factor=20.0):
    """
    Namizədin yaxınlığında ondan çox parlaq ulduz varsa True (halo/difraksiya riski).
    Belə namizəd "needs_review" olaraq işarələnir.
    """
    img = np.asarray(image, float)
    yy, xx = np.indices(img.shape)
    r = np.hypot(xx - x, yy - y)
    bkg = np.nanmedian(img[(r > radius) & (r <= 2 * radius)])
    cand = np.nanmax(img[r <= core]) - bkg
    around = img[(r > core) & (r <= radius)]
    if around.size == 0 or cand <= 0:
        return False
    return bool(np.nanmax(around) - bkg > bright_factor * cand)


def match_known(x, y, known, radius=8.0):
    """known: [(ad, x, y), ...] (SIMBAD obyektləri, piksellə). Uyğun ad və ya ''."""
    best, best_d = "", radius
    for name, kx, ky in known:
        d = np.hypot(x - kx, y - ky)
        if d <= best_d:
            best, best_d = name, d
    return best


# ---------- Əsas funksiya ----------
def analyze_candidate(img1, img2, c1, c2, hdr1, hdr2, candidate_id="cand_001",
                      survey1="DSS1 Red", survey2="DSS2 IR", spectrum=None,
                      wcs2=None, known=(), data_source="skyview_dss",
                      fallback_years=None, snr_min=5.0, absent_max=3.0):
    """
    img1, img2 : eyni piksel şəbəkəsinə gətirilmiş 2D massivlər (Epoch 1, Epoch 2)
    c1, c2     : namizədin (x, y) mərkəzi Epoch 1 və Epoch 2-də (backend tapır)
    hdr1, hdr2 : FITS başlıqları (hdul[0].header)
    spectrum   : (dalga_uzunluğu, parlaqlıq, xəta) və ya None
    wcs2       : astropy WCS (verilərsə RA/DEC hesablanır)
    fallback_years : başlıqda tarix tapılmasa istifadə olunacaq il fərqi
    """
    d1, d2 = parse_date_obs(hdr1.get("DATE-OBS")), parse_date_obs(hdr2.get("DATE-OBS"))
    if d1 and d2:
        years = (d2 - d1).days / 365.25
    else:
        years = fallback_years          # tarix tapılmadı: nəticə nəzərdən keçirilməlidir
    scale = pixel_scale_arcsec(hdr2) or pixel_scale_arcsec(hdr1)

    motion_px = float(np.hypot(c2[0] - c1[0], c2[1] - c1[1]))
    motion_as = motion_px * scale if scale else float("nan")
    pm = motion_as / years if years else float("nan")

    # "Epoch 1-də var/Epoch 2-də yox" və tərsi: obyekt hər epoxda yalnız öz yerində görünməlidir
    s1_here, s2_here = snr_peak(img1, *c1), snr_peak(img2, *c2)
    s1_there, s2_there = snr_peak(img1, *c2), snr_peak(img2, *c1)
    pair_ok = (s1_here >= snr_min and s2_here >= snr_min
               and s1_there < absent_max and s2_there < absent_max)
    halo = near_bright_star(img2, *c2) or near_bright_star(img1, *c1)

    ra = dec = ""
    if wcs2 is not None:
        ra, dec = (float(v) for v in wcs2.all_pix2world(c2[0], c2[1], 0))

    cls, conf = "", ""
    if spectrum is not None:
        res = classify_candidate(*spectrum)
        cls, conf = res["class"], res["confidence"]

    ok = pair_ok and not halo and bool(d1 and d2)
    return {
        "candidate_id": candidate_id, "ra_deg": ra, "dec_deg": dec,
        "survey_epoch1": survey1, "survey_epoch2": survey2,
        "date_obs1": d1.isoformat() if d1 else "", "date_obs2": d2.isoformat() if d2 else "",
        "pixel_scale_arcsec": round(scale, 4) if scale else "",
        "centroid1_x": round(c1[0], 2), "centroid1_y": round(c1[1], 2),
        "centroid2_x": round(c2[0], 2), "centroid2_y": round(c2[1], 2),
        "motion_arcsec": round(motion_as, 2), "pm_arcsec_yr": round(pm, 3),
        "snr": round(s2_here, 1),
        "spectral_class": cls, "spectral_source": "model", "confidence": conf,
        "data_source": data_source,
        "known_match": match_known(c2[0], c2[1], known) if known else "",
        "review_status": "auto_pass" if ok else "needs_review",
    }


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)


def evaluate_field(rows_for_report):
    """rows_for_report: [(sahə_adı, tapılan_xy, həqiqi_xy, snr_siyahısı), ...]"""
    try:
        from .validate import report
    except ImportError:
        from validate import report
    report(rows_for_report)


def pm_position(ra_deg, dec_deg, pmra_mas_yr, pmdec_mas_yr, year_obs, year_ref=2000.0):
    """
    SIMBAD-dakı məxsusi hərəkətlə ulduzun çəkiliş ilindəki RA/DEC-ni verir.
    pmra: SIMBAD-ın verdiyi qiymət (cos(dec) artıq daxildir). Nəticəni WCS ilə piksellərə çevir.
    """
    dt = year_obs - year_ref
    dra = pmra_mas_yr / 3.6e6 / np.cos(np.radians(dec_deg)) * dt
    ddec = pmdec_mas_yr / 3.6e6 * dt
    return ra_deg + dra, dec_deg + ddec


# ---------- Sınaq (süni kadrlarla, real data deyil) ----------
def _star(img, x, y, amp):
    yy, xx = np.indices(img.shape)
    img += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.5 ** 2))


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    i1, i2 = rng.normal(100, 5, (600, 600)), rng.normal(100, 5, (600, 600))
    _star(i1, 200, 200, 150)      # Epoch 1-də burada
    _star(i2, 260, 230, 150)      # Epoch 2-də burada (hərəkət edib)
    h1 = {"DATE-OBS": "23/09/53", "CDELT2": 1.5 / 3600}
    h2 = {"DATE-OBS": "1996-11-02", "CDELT2": 1.5 / 3600}
    row = analyze_candidate(i1, i2, (200, 200), (260, 230), h1, h2,
                            spectrum=None, data_source="synthetic")
    for k, v in row.items():
        print(f"{k:20s} {v}")
    print("\nGözlənən: motion ≈ 100.6\", il ≈ 43.1, pm ≈ 2.33 \"/il, auto_pass")

    # Sabit ulduz (hərəkət etməyib) -> needs_review olmalıdır
    _star(i1, 400, 400, 150); _star(i2, 400, 400, 150)
    row2 = analyze_candidate(i1, i2, (400, 400), (400, 400), h1, h2, data_source="synthetic")
    print("Sabit ulduz ->", row2["review_status"], "| motion:", row2["motion_arcsec"])
    write_csv([row, row2], "test_candidates.csv")
