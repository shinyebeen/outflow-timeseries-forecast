import pandas as pd 

df = pd.read_excel('C:\outflow\추세분석_파주.xlsx')

for column in df.columns:
    if pd.api.types.is_datetime64_any_dtype(df[column]):
        print(f'#1 {column}')
        df.rename(columns={column: 'logTime'}, inplace=True)
        break

    if pd.api.types.is_string_dtype(df[column]):
        try:
            for i in df[column]:
                if '24:00' in str(i):
                    print(i)
                    break
            print(f'#2 {column}')
            df[column] = pd.to_datetime(df[column], errors='coerce', infer_datetime_format=True)
            df.rename(columns={column: 'logTime'}, inplace=True)
            break
        except Exception as e:
            print(f"변환 실패: {e}")
            continue

print(df.info())