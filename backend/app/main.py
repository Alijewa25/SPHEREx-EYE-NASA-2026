from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.app.ai.detector import SPHERExDetector

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


@app.get("/")
def health_check():
    return {"status": "online", "system": "SPHEREx-EYE Neural Pipeline", "bands": 102}


@app.get("/api/scan/demo")
def scan_demo_sky(seed: int = 42):
    """
    Runs AI Computer Vision & 102-band Infrared Spectral Analysis on simulated multi-epoch SPHEREx data.
    """
    epoch1, epoch2 = detector.generate_synthetic_sky(seed=seed)
    return detector.detect_anomalies(epoch1, epoch2)


@app.post("/api/scan/upload")
async def scan_uploaded_epochs(
    epoch1_file: UploadFile = File(...),
    epoch2_file: UploadFile = File(...)
):
    """
    Accepts two real astronomical FITS (.fits/.fit) or image (.png/.jpg) files from Epoch 1 & Epoch 2.
    """
    try:
        bytes1 = await epoch1_file.read()
        bytes2 = await epoch2_file.read()
        img1 = detector.load_image_or_fits(bytes1, epoch1_file.filename)
        img2 = detector.load_image_or_fits(bytes2, epoch2_file.filename)
        return detector.detect_anomalies(img1, img2)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))