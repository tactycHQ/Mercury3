from predictor.predict import load_data, process_data, normalize_data
import pandas as pd

#GLOBAL VARIABLES
data_path = "D:\\Dropbox\\9. Data\\Mercury Data\\XLS\\predict\\CIQ_AAPL_predict.csv"

def main():
    df = load_data(data_path)
    norm_data = normalize_data(df)

if __name__ =='__main__':
    main()