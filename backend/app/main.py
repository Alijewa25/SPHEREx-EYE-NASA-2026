import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from backend.app.ai.detector import SPHERExDetector
from backend.app.ai.nasa_fetcher import NASADataFetcher

app = FastAPI(
    title="SPHEREx-EYE AI Mission Control API",
    description="Multi-Spectral Deep-Sky Planet X & Anomaly Hunter (NASA Space Apps 2026 - Team HERschel)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = SPHERExDetector()
nasa_fetcher = NASADataFetcher()


@app.get("/")
def serve_mission_control():
    index_path = os.path.join("frontend", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "online", "system": "SPHEREx-EYE Neural Pipeline", "bands": 102}


@app.get("/api/health")
def health_check():
    return {"status": "online", "system": "SPHEREx-EYE Neural Pipeline", "bands": 102, "data_mode": "REAL_NASA_FITS"}


@app.get("/api/scan/nasa-live")
def scan_real_nasa_data(target: str = "barnard_star"):
    try:
        e1_data = nasa_fetcher.fetch_nasa_fits(target_key=target, survey="WISE 3.4")
        e2_data = nasa_fetcher.fetch_nasa_fits(target_key=target, survey="WISE 4.6")

        result = detector.detect_anomalies(e1_data["image"], e2_data["image"])
        result["data_source"] = "NASA Goddard SkyView Real FITS Archive"
        result["target_key"] = target
        result["target_info"] = e1_data["target_name"]
        result["ra"] = e1_data["ra"]
        result["dec"] = e1_data["dec"]
        result["fits_metadata"] = {
            "epoch1": e1_data["fits_headers"],
            "epoch2": e2_data["fits_headers"],
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