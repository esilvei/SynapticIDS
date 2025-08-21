from fastapi import FastAPI

app = FastAPI(title="SynapticIDS API")


@app.get("/")
def read_root():
    return {"message": "SynapticIDS API is running"}
