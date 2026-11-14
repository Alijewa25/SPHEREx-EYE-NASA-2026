from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from backend.app.ai.detector import SPHERExDetector
from backend.app.ai.nasa_fetcher import fetch_target_epochs, TARGETS, EPOCH_MODES

app = FastAPI(
    title="SPHEREx-EYE AI Mission Control API",
    description="Multi-Spectral Deep-Sky Planet X & Anomaly Hunter (NASA Space Apps 2026 - Team HERschel)",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = SPHERExDetector()


@app.get("/api/health")
def health_check():
    return {"status": "online", "system": "SPHEREx-EYE Neural Pipeline", "bands": 102, "data_mode": "REAL_NASA_FITS"}


@app.get("/api/targets")
def get_targets():
    return {"targets": TARGETS, "epoch_modes": EPOCH_MODES}


@app.get("/api/scan/nasa-live")
def scan_real_nasa_data(target: str = "barnard_star", mode: str = "classic_1953_1996"):
    try:
        bundle = fetch_target_epochs(target_id=target, epoch_mode=mode)
        m = bundle["mode"]
        t = bundle["target"]
        meta1 = bundle["meta1"]
        meta2 = bundle["meta2"]

        result = detector.detect_anomalies(bundle["epoch1"], bundle["epoch2"])
        result["gaia_dr3"] = bundle["gaia_dr3"]
        result["data_source"] = f"NASA Goddard SkyView // {m['label']}"
        if "SYNTHETIC" in meta1.get("survey", "").upper() or "SYNTHETIC" in meta2.get("survey", "").upper():
            result["data_source"] = f"Calibrated Offline Fallback // {m['label']}"
        result["epoch_mode"] = mode
        result["epoch1_survey"] = f"{m['survey1']} ({m['epoch1_year']})"
        result["epoch2_survey"] = f"{m['survey2']} ({m['epoch2_year']})"
        result["baseline"] = m["baseline"]
        result["target_key"] = target
        result["target_info"] = t["name"]
        result["ra"] = t["ra"]
        result["dec"] = t["dec"]
        result["fits_metadata"] = {
            "epoch1": {
                "TELESCOP": "NASA-SKYVIEW",
                "INSTRUME": meta1.get("survey", m["survey1"]),
                "DATE-OBS": m["epoch1_year"],
                "CRVAL1": meta1["crval1"],
                "CRVAL2": meta1["crval2"],
            },
            "epoch2": {
                "TELESCOP": "NASA-SKYVIEW",
                "INSTRUME": meta2.get("survey", m["survey2"]),
                "DATE-OBS": m["epoch2_year"],
                "CRVAL1": meta2["crval1"],
                "CRVAL2": meta2["crval2"],
            },
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"NASA Archive fetch error: {str(e)}")


@app.post("/api/scan/upload")
async def scan_uploaded_epochs(
    epoch1_file: UploadFile = File(...),
    epoch2_file: UploadFile = File(...)
):
    try:
        bytes1 = await epoch1_file.read()
        bytes2 = await epoch2_file.read()
        img1 = detector.load_image_or_fits(bytes1, epoch1_file.filename)
        img2 = detector.load_image_or_fits(bytes2, epoch2_file.filename)
        res = detector.detect_anomalies(img1, img2)
        res["data_source"] = f"Custom FITS Plate ({epoch1_file.filename} vs {epoch2_file.filename})"
        res["target_info"] = f"Custom Observation: {epoch1_file.filename}"
        res["ra"] = 0.0
        res["dec"] = 0.0
        res["fits_metadata"] = {
            "epoch1": {"TELESCOP": "USER-FITS", "INSTRUME": epoch1_file.filename, "DATE-OBS": "CUSTOM-E1", "CRVAL1": "N/A", "CRVAL2": "N/A"},
            "epoch2": {"TELESCOP": "USER-FITS", "INSTRUME": epoch2_file.filename, "DATE-OBS": "CUSTOM-E2", "CRVAL1": "N/A", "CRVAL2": "N/A"},
        }
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend"
LOGO_PATH = BASE_DIR / "logo.png"


@app.get("/logo.png", include_in_schema=False)
def serve_logo():
    if not LOGO_PATH.is_file():
        raise HTTPException(status_code=404, detail="Logo file not found")
    return FileResponse(LOGO_PATH)


# Frontend statik faylları bütün /api/... route-larından ƏN SONDA olmalıdır
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")