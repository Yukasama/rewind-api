from dotenv import load_dotenv
import os
import fmpsdk
import pandas as pd

load_dotenv()
apikey = os.environ.get("API_KEY")

def get_history(symbol: str, from_date=None, to_date=None):
    data = fmpsdk.historical_price_full(apikey=apikey, symbol=symbol, from_date=from_date, to_date=to_date)
    return pd.DataFrame(data)

