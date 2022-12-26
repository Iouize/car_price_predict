from typing import Union

from fastapi import FastAPI

import pandas as pd
import pickle
import numpy as np

app = FastAPI()

@app.get("/")
async def root(enginesize: Union[float, None]= 150, curbweight: Union[float, None]=2000, horsepower: Union[float, None]=104):
    X_user = pd.DataFrame({
        "enginesize": [float(enginesize)],
        "curbweight": [float(curbweight)],
        "horsepower" : [float(horsepower)]
    })
    pickle_model = pickle.load(open('car_price.pkl', 'rb'))
    pred = pickle_model.predict(X_user)
    return {"car price prediction": f"{round(pred[0])}"}
