import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from run_bot import DialogueControl
from fastapi.middleware.cors import CORSMiddleware
from setfit import SetFitModel
from common import MODEL_SAVE_PATH

class Query(BaseModel):
    text: str

app = FastAPI(title="Conversational Bot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

nlu_model = None

def get_nlu_model():
    global nlu_model
    if nlu_model is None:
        nlu_model = SetFitModel.from_pretrained(MODEL_SAVE_PATH, fix_mistral_regex=True)
        print(f"Wczytano wytrenowany model z {MODEL_SAVE_PATH}")
    return nlu_model

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[WebSocket, DialogueControl] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        bot = DialogueControl(model=get_nlu_model())
        self.active_connections[websocket] = bot
        start_actions = bot.start_conversation()
        for act in start_actions:
            await websocket.send_json({
                "user": "",
                "message": act["message"],
                "display": act["display"],
                "special": act["special"]
            })

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            del self.active_connections[websocket]

    async def handle_message(self, websocket: WebSocket, message: str):
        bot = self.active_connections[websocket]
        reply_actions, intent, score = bot.process_input(message)
        for act in reply_actions:
            await websocket.send_json({
                "user": message,
                "message": act["message"],
                "display": act["display"],
                "special": act["special"]
            })

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.handle_message(websocket, data)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/predict")
async def predict(query: Query):
    if not query.text:
        raise HTTPException(status_code=400, detail="Brak tekstu do analizy")

    model = get_nlu_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model nie został zainicjalizowany")

    probs = model.predict_proba([query.text])[0]
    intent = model.predict([query.text])[0]
    score = float(probs.max())

    return {
        "intent": intent,
        "confidence": score
    }

def start_server():
    print("--- Uruchamianie serwera AI na porcie 8000 ---")
    uvicorn.run(app, host="0.0.0.0", port=8000)
