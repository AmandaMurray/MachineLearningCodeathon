import pandas as pd

def load_happy_data():
    csv_path = "world-happiness-report/2016.csv"
    return pd.read_csv(csv_path)

def main():
    happy = load_happy_data()
    happy.head()
    happy.info()

main()
