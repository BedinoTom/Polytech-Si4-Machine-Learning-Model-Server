from fastapi import FastAPI
from pydantic import BaseModel
import json
import os

app = FastAPI()

class Model(BaseModel):
    name: str
    id: str
    
class RequestImage(BaseModel):
    model_id: str
    image_raw: str
    
class ResponseImage(BaseModel):
    model_id: str
    number_class: str
    
def get_model_list():
    fp = open(os.path.join(os.getcwd(), "app", "models.json"))
    data = json.load(fp)
    models = data["models"]
    result = []
    for model in models:
        result.append(Model(name=model["name"], id=model["id"]))
    return result

def get_model_by_id(id):
    models = get_model_list()
    for model in models:
        if model.id == id:
            return model
    return None

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/proccess")
async def proccess(req: RequestImage) -> ResponseImage:
    model = get_model_by_id(req.model_id)
    if model == None:
        return ResponseImage(model_id="none", number_class="-1")

@app.get("/list")
async def model_list() -> list[Model]:
    return get_model_list()