from typing import Iterable
from fastapi import FastAPI
from dotenv import load_dotenv
import fmpsdk
import os
from stock_indicators import indicators, Quote
import logging

app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

load_dotenv()
apikey = os.environ.get("API_KEY")

@app.get("/")
async def root():
    data: Iterable[Quote] = fmpsdk.historical_price_full(apikey, symbol="EURUSD")
    logger.debug('/: data: %s', data[0])
    sma = indicators.get_sma(data, 50)
    return sma