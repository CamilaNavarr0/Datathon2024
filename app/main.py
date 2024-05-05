from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd
from io import BytesIO
import logging
from model import predict_flights, predict_sales, optimize_data, predict

app = FastAPI()

logging.basicConfig(level=logging.INFO)

@app.get("/")
async def root():
    return {"message": "Hello, this is a flight passengers predicting API. Send a POST request to /predict/ with a csv file."}

@app.post("/predict-flights")
async def upload_predict_flights(file: UploadFile = File(...)):
    try:
        input_df = pd.read_csv(BytesIO(await file.read()))
        output_df = predict_flights(input_df)
        buffer = BytesIO()
        output_df.to_csv(buffer, index=False)
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="text/csv", headers={"Content-Disposition": "attachment;filename=predicted_flights.csv"})
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing your file.")
    

@app.post("/predict-sales")
async def upload_predict_sales(file: UploadFile = File(...)):
    try:
        input_df = pd.read_csv(BytesIO(await file.read()))
        output_df = predict_sales(input_df)
        buffer = BytesIO()
        output_df.to_csv(buffer, index=False)
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="text/csv", headers={"Content-Disposition": "attachment;filename=predicted_sales.csv"})
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing your file.")
    

@app.post("/predict")
async def upload_predict(file: UploadFile = File(...)):
    try:
        input_df = pd.read_csv(BytesIO(await file.read()))
        output_df = predict(input_df)
        buffer = BytesIO()
        output_df.to_csv(buffer, index=False)
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="text/csv", headers={"Content-Disposition": "attachment;filename=predicted_data.csv"})
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing your file.")

@app.post("/optimize")
async def upload_optimize(file: UploadFile = File(...)):
    try:
        input_df = pd.read_csv(BytesIO(await file.read()))
        output_df = optimize_data(input_df)
        buffer = BytesIO()
        output_df.to_csv(buffer, index=False)
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="text/csv", headers={"Content-Disposition": "attachment;filename=optimized.csv"})
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing your file.")


@app.get("/model/{category}/{model_id}/{model_version}")
async def model(category: str, model_id: str, model_version: int):
    return {"category": category, "model_id": model_id, "model_version": model_version}
