from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.staticfiles import StaticFiles
from .schemas import InferenceResponse
from .inference import run_inference
import os

app = FastAPI()

# Mount static files directory
STATIC_DIR = "static"
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.post("/predict", response_model=InferenceResponse)
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded")

    file_bytes = await file.read()
    is_dicom = file.content_type == "application/dicom" or file.filename.endswith(".dcm")

    try:
        prediction, heatmap_base64 = run_inference(file_bytes, is_dicom=is_dicom)

        # Save heatmap to static directory
        heatmap_path = os.path.join(STATIC_DIR, f"{file.filename}_heatmap.png")
        with open(heatmap_path, "wb") as f:
            f.write(base64.b64decode(heatmap_base64))

        return InferenceResponse(
            prediction=prediction,
            heatmap_url=f"/static/{file.filename}_heatmap.png"
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

import base64
