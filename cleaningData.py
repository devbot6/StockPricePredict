import pandas as pd

# Load the original CSV
df = pd.read_csv("TSLA_historical_data.csv")

# Drop rows and columns with null values
df_cleaned = df.dropna(how='any', axis=0)  # Drop rows with null values
df_cleaned = df_cleaned.dropna(how='any', axis=1)  # Drop columns with null values

# Save the cleaned DataFrame to a new CSV
df_cleaned.to_csv('cleaned_HistoricalData_file2.csv', index=False)
