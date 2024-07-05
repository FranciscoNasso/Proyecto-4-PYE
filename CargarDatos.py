try:
    import pandas as pd
except ImportError:
    print("Please install pandas library.")

def cargarDB(archivo : str) -> pd.DataFrame:
    return pd.read_csv(archivo,delimiter=";",header=0)