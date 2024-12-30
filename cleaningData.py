import pandas as pd

# Load the original CSV
df = pd.read_csv("historical_stock_data.csv")

# Keep only the first 3 columns
df_cleaned = df.iloc[:, :3]

finalClean = df_cleaned.dropna()

# Save the cleaned DataFrame to a new CSV
finalClean.to_csv('cleaned_HistoricalData_file.csv', index=False)
