import pandas as pd


def parse_date(date_string):
    return pd.datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")


def read_relevant_data():
    df = pd.read_csv("../697_data/yellow_tripdata_2015-09.csv", header=0, usecols=["tpep_pickup_datetime",
                                                                                   "tpep_dropoff_datetime",
                                                                                   "pickup_longitude",
                                                                                   "pickup_latitude",
                                                                                   "dropoff_longitude",
                                                                                   "dropoff_latitude"],
                     parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"],
                     date_parser=parse_date, nrows=351276,
                     dtype={"pickup_longitude": "float64", "pickup_latitude": "float64", "dropoff_longitude": "float64",
                            "dropoff_latitude": "float64"})

    # df.info()
    return df


def remove_rows_that_contain_0_values(df):
    return df[(df['pickup_longitude'] != float(0)) & (df['pickup_latitude'] != float(0)) &
              (df['dropoff_longitude'] != float(0)) & (df['dropoff_latitude'] != float(0))]


def remove_noisy_rows(df):
    return df[(df['pickup_longitude'] < float(-70)) & (df['pickup_latitude'] > float(40)) &
              (df['dropoff_longitude'] < float(-70)) & (df['dropoff_latitude'] > float(40))]


def impose_boundary(df):
    return df[(df['pickup_longitude'] <= float(-73.7)) & (df['pickup_longitude'] >= float(-74.2)) &
              (df['pickup_latitude'] >= float(40.5)) & (df['pickup_latitude'] <= float(41)) &
              (df['dropoff_longitude'] <= float(-73.7)) & (df['dropoff_longitude'] >= float(-74.2)) &
              (df['dropoff_latitude'] >= float(40.5)) & (df['dropoff_latitude'] <= float(41))]


def sample_data(df):
    sample_size = 100000
    if len(df) > sample_size:
        df = df.sample(sample_size)

    return df


def get_hourly_data(df, data_type):
    hourly_df_list = []
    for i in range(0, 24):

        if data_type == 'pickup':
            hourly_df = df[((df['tpep_pickup_datetime'].dt.hour >= i) & (df['tpep_pickup_datetime'].dt.hour < (i + 1)))]
            hourly_df = hourly_df[['pickup_longitude', 'pickup_latitude']]
            hourly_df = hourly_df.rename(columns={"pickup_longitude": "longitude", "pickup_latitude": "latitude"})

        elif data_type == 'dropoff':
            hourly_df = df[((df['tpep_dropoff_datetime'].dt.hour >= i) & (df['tpep_dropoff_datetime'].dt.hour < (i + 1)))]
            hourly_df = hourly_df[['dropoff_longitude', 'dropoff_latitude']]
            hourly_df = hourly_df.rename(columns={"dropoff_longitude": "longitude", "dropoff_latitude": "latitude"})

        else:
            return []

        hourly_df_list.append(hourly_df)

        print(len(hourly_df))

    return hourly_df_list


def get_whole_specific_data(df, data_type):
    specific_df = None

    if data_type == 'pickup':
        specific_df = df[['pickup_longitude', 'pickup_latitude']]
        specific_df = specific_df.rename(columns={"pickup_longitude": "longitude", "pickup_latitude": "latitude"})

    elif data_type == 'dropoff':
        specific_df = df[['dropoff_longitude', 'dropoff_latitude']]
        specific_df = specific_df.rename(columns={"dropoff_longitude": "longitude", "dropoff_latitude": "latitude"})

    return specific_df


def read_and_preprocess_data():
    df = read_relevant_data()
    # print(df.head(10))
    print("Total data: %s" % len(df))

    df = remove_rows_that_contain_0_values(df)
    print("After removing 0 value rows: %s" % len(df))
    # print(df.tail(10))

    # from the x y plot we see there are a few data points with abnormal longitude or latitude values
    # try commenting following line and see the difference in plot of dropoff
    df = remove_noisy_rows(df)
    print("After removing noisy rows: %s" % len(df))

    df = impose_boundary(df)
    print("After imposing boundary: %s" % len(df))

    return df
