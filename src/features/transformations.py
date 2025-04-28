import pandas as pd
from haversine import haversine

from src.utils.time import robust_hour_of_iso_date


def driver_distance_to_pickup(df: pd.DataFrame) -> pd.DataFrame:
    df["driver_distance"] = df.apply(
        lambda r: haversine(
            (r["driver_latitude"], r["driver_longitude"]),
            (r["pickup_latitude"], r["pickup_longitude"]),
        ),
        axis=1,
    )
    return df


def hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    df["event_hour"] = df["event_timestamp"].apply(robust_hour_of_iso_date)
    return df


def driver_historical_completed_bookings(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError(
        f"Show us your feature engineering skills! Suppose that drivers with a good track record are more likely to accept bookings. "
        f"Implement a feature that describes the number of historical bookings that each driver has completed."
    )
