import pandas as pd


def parse_date(date_string):
    return pd.datetime.strptime(date_string, "%Y-%d-%m %H:%M:%S")


def remove_rows_that_contain_0_values(df):
    return df[(df['pickup_longitude'] != float(0)) & (df['pickup_latitude'] != float(0)) &
              (df['dropoff_longitude'] != float(0)) & (df['dropoff_latitude'] != float(0))]


def read_relevant_data():
    df = pd.read_csv("../697_data/yellow_tripdata_2015-09.csv", header=0, usecols=["tpep_pickup_datetime",
                                                                            "tpep_dropoff_datetime", "pickup_longitude",
                                                                            "pickup_latitude", "dropoff_longitude",
                                                                            "dropoff_latitude"],
                     parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"],
                     date_parser=parse_date, nrows=100000,
                     dtype={"pickup_longitude": "float64", "pickup_latitude": "float64", "dropoff_longitude": "float64",
                            "dropoff_latitude": "float64"})

    # df.info()
    return df


def main():
    df = read_relevant_data()
    print(df.head(10))

    df = remove_rows_that_contain_0_values(df)
    print(df.tail(10))


if __name__ == "__main__":
    main()
