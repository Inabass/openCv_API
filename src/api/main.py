import GAF
import GFD
import FR
from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

app = FastAPI()

@app.post("/detect")
async def create_upload_files(image: UploadFile = File(...)):
        filepath = save_upload_file_tmp(image)
        try:
             res = GAF.main(filepath)
             GFD.main()
        except Exception as e:
            raise HTTPException(status_code=500, detail='Failed to convert an image.')
        finally:
            filepath.unlink()
        # print(filepath)
        
        # data = await image.read()

        # res = GAF.main(data)
        # if res is False:
        #     return {"message": "failed"}
            
        # if res is True:
        #     GFD.main()
        #     return {"message": "succes"}
        

@app.post("/recognize")
async def create_upload_files(image: UploadFile  = File(...)):
        filepath = save_upload_file_tmp(image)
        try:
             FR.main(filepath)
        except Exception as e:
            raise HTTPException(status_code=500, detail='Failed to convert an image.')
        finally:
            filepath.unlink()

def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
        tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()
    return tmp_path