from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# Using your local saved fine-tuned model path
MODEL_ID = "/content/Vanessa-and-Jimia_model"

app = FastAPI(title="Sentiment API")

# Load the model ONCE when the app starts
clf = pipeline("text-classification", model=MODEL_ID)

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    label: str
    score: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOut)
def predict(item: TextIn):
    if not item.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    result = clf(item.text)[0]
    return PredictionOut(label=result["label"], score=round(result["score"], 4))
