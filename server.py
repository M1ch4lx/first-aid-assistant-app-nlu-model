import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from run_bot import FirstAidBot
from fastapi.middleware.cors import CORSMiddleware

class Query(BaseModel):
    text: str

app = FastAPI(title="First Aid Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bot = None

def get_bot():
    global bot
    if bot is None:
        bot = FirstAidBot()
    return bot

@app.post("/predict")
async def predict(query: Query):
    if not query.text:
        raise HTTPException(status_code=400, detail="Brak tekstu do analizy")
    
    current_bot = get_bot()
    if not current_bot.model:
        raise HTTPException(status_code=500, detail="Model nie został zainicjalizowany")
    
    intent, score = current_bot.get_intent(query.text)
    return {
        "intent": intent,
        "confidence": float(score)
    }

def start_server():
    print("--- Uruchamianie serwera AI na porcie 8000 ---")
    uvicorn.run(app, host="0.0.0.0", port=8000)