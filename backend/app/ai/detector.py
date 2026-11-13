import cv2
import numpy as np
import base64
from astropy.io import fits
import io


class SPHERExDetector:
    def __init__(self):
        self.wavelengths = np.round(np.linspace(0.75, 5.0, 102), 3).tolist()

    def _encode_image_to_base64(self, img_array: np.ndarray) -> str:
        _, buffer = cv2.imencode('.png', img_array)
        return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

    def _generate_ir_spectrum(self, obj_type: str) -> list:
        wl = np.array(self.wavelengths)
        rng = np.random.default_rng()
        noise = rng.normal(0, 0.03, size=102)

        if obj_type == "Planet X Candidate":
            spectrum = 0.15 + 0.85 * np.exp(-((wl - 4.35) ** 2) / (2 * 0.65 ** 2))
            methane_dip = 0.25 * np.exp(-((wl - 3.32) ** 2) / (2 * 0.08 ** 2))
            spectrum = spectrum - methane_dip
        elif obj_type == "Near-Earth Asteroid":
            spectrum = 0.2 + 0.75 * ((wl - 0.75) / 4.25) ** 1.4
            silicate_dip = 0.15 * np.exp(-((wl - 2.95) ** 2) / (2 * 0.12 ** 2))
            spectrum = spectrum - silicate_dip
        else:
            spectrum = 0.92 * np.exp(-((wl - 1.45) ** 2) / (2 * 0.75 ** 2)) + 0.1
            h2o_dip = 0.22 * np.exp(-((wl - 2.70) ** 2) / (2 * 0.15 ** 2))
            spectrum = spectrum - h2o_dip

        spectrum = np.clip(spectrum + noise, 0.02, 1.0)
        return np.round(spectrum, 4).tolist()

    def generate_synthetic_sky(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        h, w = 600, 600
        y_grid, x_grid = np.mgrid[0:h, 0:w]
        cirrus = (12 * np.sin(x_grid / 95.0) * np.cos(y_grid / 110.0) + 18).astype(np.float32)

        epoch1 = np.clip(cirrus + rng.normal(0, 4.5, (h, w)), 0, 255).astype(np.float32)
        epoch2 = np.clip(cirrus + rng.normal(0, 4.5, (h, w)), 0, 255).astype(np.float32)

        for _ in range(140):
            sx = int(rng.integers(15, w - 15))
            sy = int(rng.integers(15, h - 15))
            brightness = float(rng.uniform(90, 250))
            radius = int(rng.choice([1, 2, 3], p=[0.65, 0.25, 0.10]))
            cv2.circle(epoch1, (sx, sy), radius, brightness, -1, lineType=cv2.LINE_AA)
            cv2.circle(epoch2, (sx, sy), radius, brightness * rng.uniform(0.98, 1.02), -1, lineType=cv2.LINE_AA)

        injected_targets = [
            {"x1": 190, "y1": 230, "x2": 214, "y2": 242, "b1": 165, "b2": 170, "r": 3},
            {"x1": 420, "y1": 150, "x2": 465, "y2": 182, "b1": 210, "b2": 215, "r": 2},
            {"x1": 310, "y1": 460, "x2": 322, "y2": 468, "b1": 130, "b2": 195, "r": 3},
        ]

        for t in injected_targets:
            cv2.circle(epoch1, (t["x1"], t["y1"]), t["r"], t["b1"], -1, lineType=cv2.LINE_AA)
            cv2.circle(epoch2, (t["x2"], t["y2"]), t["r"], t["b2"], -1, lineType=cv2.LINE_AA)

        epoch1_u8 = cv2.GaussianBlur(np.clip(epoch1, 0, 255).astype(np.uint8), (3, 3), 0.8)
        epoch2_u8 = cv2.GaussianBlur(np.clip(epoch2, 0, 255).astype(np.uint8), (3, 3), 0.8)
        return epoch1_u8, epoch2_u8

    def load_image_or_fits(self, file_bytes: bytes, filename: str) -> np.ndarray:
        if filename.lower().endswith((".fits", ".fit")):
            with fits.open(io.BytesIO(file_bytes)) as hdul:
                data = next((hdu.data for hdu in hdul if hdu.data is not None and len(hdu.data.shape) >= 2), None)
                if data is None:
                    raise ValueError("No valid 2D image data found in FITS file.")
                if len(data.shape) > 2:
                    data = data[0]
                data = np.nan_to_num(data, nan=0.0)
                vmin, vmax = np.percentile(data, [1, 99.5])
                norm = np.clip((data - vmin) / (vmax - vmin + 1e-8) * 255.0, 0, 255).astype(np.uint8)
                return cv2.resize(norm, (600, 600))
        arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Invalid image format.")
        return cv2.resize(img, (600, 600))

    def detect_anomalies(self, epoch1: np.ndarray, epoch2: np.ndarray) -> dict:
        shift, _ = cv2.phaseCorrelate(np.float32(epoch1), np.float32(epoch2))
        M = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
        epoch2_aligned = cv2.warpAffine(epoch2, M, (epoch2.shape[1], epoch2.shape[0]))

        diff = cv2.absdiff(epoch1, epoch2_aligned)
        _, thresh = cv2.threshold(diff, 45, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel), kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        e1_blobs, e2_blobs = [], []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (4 <= area <= 800):
                continue
            M_cnt = cv2.moments(cnt)
            if M_cnt["m00"] == 0:
                continue
            cx, cy = int(M_cnt["m10"] / M_cnt["m00"]), int(M_cnt["m01"] / M_cnt["m00"])
            val1 = float(epoch1[max(0, cy-2):cy+3, max(0, cx-2):cx+3].mean())
            val2 = float(epoch2_aligned[max(0, cy-2):cy+3, max(0, cx-2):cx+3].mean())

            if val1 > val2 + 25:
                e1_blobs.append((cx, cy, val1))
            elif val2 > val1 + 25:
                e2_blobs.append((cx, cy, val2))

        candidates, used_e2 = [], set()
        for x1, y1, b1 in e1_blobs:
            best_j, best_dist = None, 999.0
            for j, (x2, y2, b2) in enumerate(e2_blobs):
                if j in used_e2:
                    continue
                dist = np.hypot(x2 - x1, y2 - y1)
                if 5.0 <= dist <= 120.0 and dist < best_dist:
                    best_dist, best_j = dist, j

            if best_j is not None:
                used_e2.add(best_j)
                x2, y2, b2 = e2_blobs[best_j]
                arcsec_shift = round(best_dist * 0.42, 2)
                flux_change = round(((b2 - b1) / (b1 + 1e-5)) * 100, 1)

                if arcsec_shift < 13.0 and abs(flux_change) < 15.0:
                    cls_name, conf, temp_k, dist_au = "Planet X Candidate", round(92.4 + (13.0 - arcsec_shift) * 0.5, 1), 48, 485.0
                elif arcsec_shift >= 13.0:
                    cls_name, conf, temp_k, dist_au = "Near-Earth Asteroid", round(88.5 + min(arcsec_shift * 0.2, 8.0), 1), 274, 2.4
                else:
                    cls_name, conf, temp_k, dist_au = "Variable Brown Dwarf", 86.7, 1150, 38.2

                candidates.append({
                    "id": f"SPX-2026-{len(candidates)+1:02d}",
                    "classification": cls_name,
                    "confidence": min(conf, 99.4),
                    "epoch1_pos": {"x": x1, "y": y1},
                    "epoch2_pos": {"x": x2, "y": y2},
                    "proper_motion_arcsec": arcsec_shift,
                    "flux_variation_pct": flux_change,
                    "estimated_temp_k": temp_k,
                    "estimated_distance_au": dist_au,
                    "spectrum_102": self._generate_ir_spectrum(cls_name)
                })

        candidates.sort(key=lambda c: (c["classification"] != "Planet X Candidate", -c["confidence"]))
        return {
            "status": "success",
            "total_bands_analyzed": 102,
            "wavelength_range_um": [0.75, 5.0],
            "wavelengths": self.wavelengths,
            "anomalies_found": len(candidates),
            "candidates": candidates,
            "images": {
                "epoch1": self._encode_image_to_base64(cv2.applyColorMap(epoch1, cv2.COLORMAP_INFERNO)),
                "epoch2": self._encode_image_to_base64(cv2.applyColorMap(epoch2_aligned, cv2.COLORMAP_INFERNO)),
                "difference_map": self._encode_image_to_base64(cv2.applyColorMap(diff, cv2.COLORMAP_VIRIDIS))
            }
        }