from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.app.ai.detector import SPHERExDetector
from backend.app.ai.nasa_fetcher import NASADataFetcher

app = FastAPI(
    title="SPHEREx-EYE AI API",
    description="Multi-Spectral Deep-Sky Planet X & Anomaly Hunter (NASA Space Apps 2026 - Team HERschel)",
    version="1.0.0"
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
def health_check():
    return {"status": "online", "system": "SPHEREx-EYE Neural Pipeline", "bands": 102}


@app.get("/api/scan/nasa-live")
def scan_real_nasa_data(target: str = "sedna_region"):
    """
    Pulls real FITS sky imagery directly from NASA SkyView/IRSA servers,
    compares multi-band infrared observations (WISE 3.4um vs WISE 4.6um),
    and runs AI anomaly detection.
    """
    try:
        e1_data = nasa_fetcher.fetch_nasa_fits(target_key=target, survey="WISE 3.4")
        e2_data = nasa_fetcher.fetch_nasa_fits(target_key=target, survey="WISE 4.6")

        result = detector.detect_anomalies(e1_data["image"], e2_data["image"])
        result["data_source"] = "NASA SkyView / IRSA Real FITS Archive"
        result["target_info"] = e1_data["target_name"]
        result["fits_metadata"] = e1_data["fits_headers"]
        return result
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"NASA Archive fetch error: {str(e)}")


@app.get("/api/scan/demo")
def scan_demo_sky(seed: int = 42):
    epoch1, epoch2 = detector.generate_synthetic_sky(seed=seed)
    res = detector.detect_anomalies(epoch1, epoch2)
    res["data_source"] = "SPHEREx 102-Band Benchmark Simulation"
    return res


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
        res["data_source"] = f"User Uploaded NASA FITS/Image ({epoch1_file.filename})"
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))