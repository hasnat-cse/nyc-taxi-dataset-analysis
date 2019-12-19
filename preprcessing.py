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
                     date_parser=parse_date, nrows=2400000,
                     dtype={"pickup_longitude": "float64", "pickup_latitude": "float64", "dropoff_longitude": "float64",
                            "dropoff_latitude": "float64"})


    # df = pd.read_csv("../697_data/yellow_tripdata_2015-09.csv", header=0, usecols=["tpep_pickup_datetime",
    #                                                                                "tpep_dropoff_datetime",
    #                                                                                "pickup_longitude",
    #                                                                                "pickup_latitude",
    #                                                                                "dropoff_longitude",
    #                                                                                "dropoff_latitude",
    #                                                                                "trip_distance"],
    #                  parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"],
    #                  date_parser=parse_date, nrows=2400000,
    #                  dtype={"pickup_longitude": "float64", "pickup_latitude": "float64", "dropoff_longitude": "float64",
    #                         "dropoff_latitude": "float64", "trip_distance": "float64"})

    return df


def remove_rows_that_contain_0_values(df):
    return df[(df['pickup_longitude'] != float(0)) & (df['pickup_latitude'] != float(0)) &
              (df['dropoff_longitude'] != float(0)) & (df['dropoff_latitude'] != float(0))]

    # return df[(df['pickup_longitude'] != float(0)) & (df['pickup_latitude'] != float(0)) &
    #           (df['dropoff_longitude'] != float(0)) & (df['dropoff_latitude'] != float(0)) &
    #           (df['trip_distance'] > float(0))]


def impose_boundary(df):
    return df[(df['pickup_longitude'] <= float(-73.6992)) & (df['pickup_longitude'] >= float(-74.2572)) &
              (df['pickup_latitude'] >= float(40.4960)) & (df['pickup_latitude'] <= float(40.9156)) &
              (df['dropoff_longitude'] <= float(-73.6992)) & (df['dropoff_longitude'] >= float(-74.2572)) &
              (df['dropoff_latitude'] >= float(40.4960)) & (df['dropoff_latitude'] <= float(40.9156))]


def sample_data(df, sample_size):
    sample_size = sample_size
    if len(df) > sample_size:
        df = df.sample(sample_size)

    return df


def get_periodic_data(df, periods):
    periodic_df_list = []

    for period in periods:
        periodic_df = df[
            ((df['tpep_pickup_datetime'].dt.hour >= period[0]) & (df['tpep_pickup_datetime'].dt.hour < period[1]))]

        periodic_df = periodic_df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']]

        # periodic_df = periodic_df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
        #                            'dropoff_latitude', 'trip_distance']]

        periodic_df_list.append(periodic_df)

    return periodic_df_list


def read_and_preprocess_data():
    df = read_relevant_data()

    df = get_data_of_one_week_only(df)

    df = remove_rows_that_contain_0_values(df)
    # print("After removing 0 value rows: %s" % len(df))

    df = impose_boundary(df)
    # print("After imposing boundary: %s" % len(df))

    weekday_df, weekend_df = divide_data_by_weekday_weekend(df)
    print("Weekday data length: %s" % len(weekday_df))
    # print(weekday_df.tail(5))

    print("Weekend data length: %s" % len(weekend_df))
    # print(weekend_df.tail(5))

    return weekday_df, weekend_df


def get_data_of_one_week_only(df):
    # get data from September 1 to 7
    df = df[(df['tpep_pickup_datetime'] < parse_date("2015-09-08 00:00:00"))]

    return df


def divide_data_by_weekday_weekend(df):
    # wednesday(2)
    weekday_df = df[(df['tpep_pickup_datetime'].dt.dayofweek == 2)]

    # sunday(6)
    weekend_df = df[(df['tpep_pickup_datetime'].dt.dayofweek == 6)]

    return weekday_df, weekend_df
