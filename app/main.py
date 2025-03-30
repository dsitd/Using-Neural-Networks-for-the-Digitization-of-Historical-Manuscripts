from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
import uvicorn

from app.predict import simple_predict, extract_sorted_text_with_newlines

MODEL_PATH = "./model/finnal_model_95.pt"

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_filename = "temp_image.jpg"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = simple_predict(
        image_path=temp_filename,
        model_path=MODEL_PATH,
        model_type='yolov8',
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        conf_threshold=0.5
    )

    text = extract_sorted_text_with_newlines(
        result,
        ignore_labels=["cherta", "dom"],
        line_threshold=15
    )

    os.remove(temp_filename)

    return JSONResponse(content={"text": text})

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=True)
