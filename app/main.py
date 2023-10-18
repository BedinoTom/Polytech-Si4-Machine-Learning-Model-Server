from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os

from . import utils
from . import inference

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    return data["models"]

def get_model_by_id(id):
    models = get_model_list()
    for model in models:
        if model["id"] == id:
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
    image_str = req.image_raw
    np_image = utils.decode_image(image_str.encode("utf-8"))
    infere_class = inference.inference(model, np_image)
    return ResponseImage(model_id=model["id"], number_class=infere_class)

@app.get("/list")
async def model_list() -> list[Model]:
    result = []
    for model in get_model_list():
        result.append(Model(name=model["name"], id=model["id"]))
    return result