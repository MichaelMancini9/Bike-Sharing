from pydantic import BaseModel, Field

class BikeInput(BaseModel):
    season: int = Field(..., ge=1, le=4)
    yr: int = Field(..., ge=0, le=1)
    mnth: int = Field(..., ge=1, le=12)
    hr: int = Field(..., ge=0, le=23)
    holiday: int = Field(..., ge=0, le=1)
    weekday: int = Field(..., ge=0, le=6)
    workingday: int = Field(..., ge=0, le=1)
    weathersit: int = Field(..., ge=1, le=4)
    temp: float
    hum: float
    windspeed: float