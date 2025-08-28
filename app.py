from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.constants import APP_HOST,APP_PORT
from src.pipline.prediction_pipeline import PredictionPipeline, SpotifyData  
from uvicorn import run as app_run
app = FastAPI()

# Static + Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    danceability: float = Form(...),
    energy: float = Form(...),
    key: int = Form(...),
    loudness: float = Form(...),
    mode: int = Form(...),
    speechiness: float = Form(...),
    acousticness: float = Form(...),
    instrumentalness: float = Form(...),
    liveness: float = Form(...),
    valence: float = Form(...),
    tempo: float = Form(...),
    duration_ms: int = Form(...),
    time_signature: int = Form(...),
    chorus_hit: float = Form(...),
    sections: int = Form(...)
):
    # Create data object
    data = SpotifyData(
        danceability, energy, key, loudness, mode,
        speechiness, acousticness, instrumentalness,
        liveness, valence, tempo, duration_ms,
        time_signature, chorus_hit, sections
    )

    # Run prediction
    pipeline = PredictionPipeline()
    prediction = pipeline.predict(data)

    result = "🎵 Likely a Hit Song!" if prediction == 1 else "❌ Not a Hit"
    print(result)
    return templates.TemplateResponse(
        "index.html", {"request": request, "result": result}
    )
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)