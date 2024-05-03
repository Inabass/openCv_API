import GAF
import GFD
import FR
from fastapi import FastAPI, UploadFile

app = FastAPI()

@app.post("/detect")
async def create_upload_files(image: UploadFile):
        data = await image.read()

        res = GAF.main(data)
        if res is False:
            return {"message": "failed"}
            
        if res is True:
            GFD.main()
            return {"message": "succes"}
        

@app.post("/recognize")
async def create_upload_files(image: UploadFile):
        data = await image.read()

        FR.main(data)

        return {"message": image.filename}