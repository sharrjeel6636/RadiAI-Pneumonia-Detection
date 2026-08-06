from pydantic import BaseModel

class InferenceResponse(BaseModel):
    prediction: str
    heatmap_url: str
