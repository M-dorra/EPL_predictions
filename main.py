from fastapi import FastAPI
from pydantic import BaseModel
from model import predict_match
from fastapi.middleware.cors import CORSMiddleware
from model import matches

app = FastAPI()
#Enable CORS for frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class MatchInput(BaseModel):
    team1: str
    team2: str

@app.get("/teams")
def get_teams():
    teams = matches["team"].dropna().unique().tolist()
    return {"teams": sorted(teams)}

@app.post("/predict")
def predict(input: MatchInput):
    prediction = predict_match(input.team1, input.team2)
    return {"prediction": prediction}
