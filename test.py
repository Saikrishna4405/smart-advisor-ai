import pandas as pd
df = pd.read_csv('dataset/india/ASIANPAINT.NS.csv')
df.columns = df.columns.str.strip()
df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
close_col = next((c for c in df.columns if c.lower() == 'close'), None)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.rename(columns={close_col: 'Close'}, inplace=True)
df = df.dropna(subset=['Date', 'Close']).sort_values('Date')
print('Orig length:', len(df))
max_date = df['Date'].max()
print('Max date:', max_date)
df_1y = df[df['Date'] >= max_date - pd.DateOffset(years=1)]
print('1Y length:', len(df_1y))
