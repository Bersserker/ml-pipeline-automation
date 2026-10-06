import pandas as pd

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
  """Базования отчистка данных"""

  df = df.copy()
  #убираем дубликаты
  df = df.drop_duplicates()

  #названия столбцов к единому фломату
  df.columns = df.columns.str.strip().str.lower().str.replace(' ','_').str.replace('.','_')

  df = df.rename(columns={'default_payment_next_month':'default'})
  #если нет целевой переменной убираем и датасета
  df = df.dropna(subset='default')

  return df