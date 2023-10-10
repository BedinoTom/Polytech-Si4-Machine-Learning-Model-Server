from fastapi import FastAPI
from pydantic import BaseModel
import json
import os

app = FastAPI()

class Model(BaseModel):
    name: str
    id: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/list")
async def model_list() -> list[Model]:
    fp = open(os.path.join(os.getcwd(), "app", "models.json"))
    data = json.load(fp)
    models = data["models"]
    result = []
    for model in models:
        result.append(Model(name=model["name"], id=model["id"]))
    return result