import joblib
import pandas as pd

from fastapi import FastAPI

from pydantic import BaseModel , Field
from typing import Literal



from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(title="Medical Cost Personal API" , description="Sağlık siogortası maliyeti hesaplayan ML model api")


# bu alttaki kısmı yazmadan önce port farklılığından dolayı api'ye bağlanamadık (CORSMiddleware import ettik)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # her porttan gelen isteğe izin ver
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





model = joblib.load("sigorta_hesapla.joblib")
model_ozellikleri = joblib.load("model_features.joblib")



class kişi_verisi(BaseModel):

    age : int = Field(ge=18 , le=64)
    sex : Literal["female","male"]
    bmi : float = Field(ge=15.96 ,le =53.13)
    children : int = Field(ge=1,le=5)
    smoker : Literal["yes", "no"]
    region : Literal["southwest","southeast","northwest","northeast"]
    
   # vki_sinif : Literal["Fazla_Kilolu","Obez_1","Normal","Obez_2","Obez_3","Zayıf"]

def vki_siniflandirma(vki):
    if vki < 18.5:
        return "Zayıf"
    elif 18.5 <= vki < 25.0:
        return "Normal"
    elif 25.0 <= vki < 30.0:
        return "Fazla_Kilolu"
    elif 30.0 <= vki < 35.0:
        return "Obez_1"
    elif 35.0 <= vki < 40.0:
        return "Obez_2"
    else:
        return "Obez_3"


@app.get("/")
def ana_sayfa():
    return {"Bilgilendirme": "Sağlık sigorta maliyeti hesaplayan fastapi uygulaması"}



@app.post("/tahmin")
def tahmin_sonucu(veri : kişi_verisi):

    # model_dump() methodu pydantic'in kendisinindir, Pydantic modelinin verilerini bir sözlük olarak döndürür. sonuç = gelen jsonu dataframe çevirdik , üzerinde işlem yapabilelim diye.

    veri_sozlugu = veri.model_dump()

    df = pd.DataFrame([veri_sozlugu])

    # kullanıcının girdiği bmi değerine göre vki_sinif'ı otomatik hesaplıyoruz
    df["vki_sinif"] = df["bmi"].apply(vki_siniflandirma)

    
    # model (Pipeline) ile Tahmin Yapma
    # (pipeline, vki_sinif ve diğer kategorik sütunları otomatik encode ediyor)
    tahmin_dizi = model.predict(df)
    tahmin_maliyet = float(tahmin_dizi[0])

    # vki_sınıf = df["vki_sinif"] --> aşağıdaki şekilde düzelltik
    vki_sınıf = df.loc[0, "vki_sinif"]
    
    # yanıt döndürme
    return {

            "tahmin_edilen_sigorta_maliyeti": round(tahmin_maliyet, 2),
            "sınıf" : vki_sınıf
    }