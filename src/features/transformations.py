import pandas as pd
from haversine import haversine
from sklearn.neighbors import BallTree
from datetime import timedelta
import numpy as np
from src.utils.time import robust_hour_of_iso_date
from geopy.distance import geodesic
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.utils.time import iso_to_datetime

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
    if not pd.api.types.is_datetime64_any_dtype(df["event_timestamp"]):
        df["event_timestamp"] = df["event_timestamp"].apply(iso_to_datetime)
    
    df['event_timestamp'] = df['event_timestamp'].dt.tz_localize(None)
    df["event_hour"] = df["event_timestamp"].dt.hour.astype(int)
    return df



def driver_historical_completed_bookings(
    df: pd.DataFrame, historical_data: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Adds a feature to the DataFrame that represents the number of historical bookings
    completed by each driver. For test data, it uses precomputed historical data.

    Args:
        df (pd.DataFrame): Input DataFrame.
        historical_data (pd.DataFrame, optional): Precomputed historical data for test data.

    Returns:
        pd.DataFrame: DataFrame with an additional column 'historical_completed_bookings'.
    """
    if historical_data is None:
        # Compute historical completed bookings from the current DataFrame (training data)
        completed_bookings = df[df["is_completed"] == 1]
        historical_counts = completed_bookings.groupby("driver_id").size().reset_index(
            name="historical_completed_bookings"
        )
    else:
        # Use precomputed historical data for test data
        historical_counts = historical_data[["driver_id", "historical_completed_bookings"]]


    historical_counts=historical_counts.drop_duplicates(subset=['driver_id'], keep='last')

    # Merge the counts back to the original DataFrame
    df = df.merge(historical_counts, on="driver_id", how="left")

    # Fill NaN values with 0 for drivers with no completed bookings
    df["historical_completed_bookings"] = df["historical_completed_bookings"].fillna(0).astype(int)
    return df




def drop_near_next_pickup(df: pd.DataFrame, historical_data: pd.DataFrame = None, radius_km=1, time_limit_min=20, avg_speed_kmph=40) -> pd.DataFrame:
    df = df.copy()
    if historical_data is None:
        historical_data = df.copy()


    historical_data = historical_data.copy()
    historical_data = historical_data.sort_values("event_timestamp")
    if not pd.api.types.is_datetime64_any_dtype(df['event_timestamp']):
        df['event_timestamp'] = df['event_timestamp'].apply(lambda x: iso_to_datetime(x))
    # Ensure event_timestamp has no timezone info
    if not pd.api.types.is_datetime64_any_dtype(historical_data['event_timestamp']):
        historical_data['event_timestamp'] = historical_data['event_timestamp'].apply(lambda x: iso_to_datetime(x))
    df['event_timestamp'] = df['event_timestamp'].dt.tz_localize(None)
    historical_data['event_timestamp'] = historical_data['event_timestamp'].dt.tz_localize(None)
    # Create driver last drop dictionary from historical_data
    driver_last_drop = {}

    for idx, row in historical_data.iterrows():
        if row.get("is_completed") == 1:
            est_duration_min = (row["trip_distance"] / avg_speed_kmph) * 60.0
            estimated_drop_time = row["event_timestamp"] + pd.Timedelta(minutes=est_duration_min)  # Use pd.Timedelta
            drop_latlon = (row["driver_latitude"], row["driver_longitude"])
            driver = row["driver_id"]
            driver_last_drop[driver] = (estimated_drop_time, drop_latlon)

    # Now process df (train/test) using the driver_last_drop
    df["drop_near_next_pickup"] = 0
    radius = radius_km / 6371  # Radius for haversine

    df = df.sort_values("event_timestamp")

    for idx, row in df.iterrows():
        driver = row["driver_id"]
        curr_ts = row["event_timestamp"]
        pickup_latlon = (row["pickup_latitude"], row["pickup_longitude"])

        # Check if last estimated drop is within time window
        if driver in driver_last_drop:
            drop_ts, drop_latlon = driver_last_drop[driver]
            time_diff = (curr_ts - drop_ts).total_seconds() / 60.0

            if time_diff <= time_limit_min:
                tree = BallTree(np.radians([drop_latlon]), metric="haversine")
                if len(tree.query_radius(np.radians([pickup_latlon]), r=radius)[0]) > 0:
                    df.at[idx, "drop_near_next_pickup"] = 1

        # If current ride is completed, update the driver's last drop
        if row.get("is_completed") == 1:
            est_duration_min = (row["trip_distance"] / avg_speed_kmph) * 60.0
            estimated_drop_time = row["event_timestamp"] + pd.Timedelta(minutes=est_duration_min)  # Use pd.Timedelta
            drop_latlon = (row["driver_latitude"], row["driver_longitude"])
            driver_last_drop[driver] = (estimated_drop_time, drop_latlon)

    return df





def add_driver_familiarity_features(df: pd.DataFrame, historical_data: pd.DataFrame = None, radius_km: float = 0.3) -> pd.DataFrame:
    df = df.sort_values('event_timestamp').copy()
    df["driver_familiar_with_pickup"] = 0
    df["driver_familiar_with_dropoff"] = 0

    earth_radius = 6371
    radius = radius_km / earth_radius

    # Prepare containers for driver histories
    driver_pickup_history = {}
    driver_dropoff_history = {}

    # If historical_data is provided, prefill histories
    if historical_data is not None:
        historical_data = historical_data.dropna(subset=["pickup_latitude", "pickup_longitude", "driver_latitude", "driver_longitude"])
        historical_data = historical_data[historical_data["is_completed"] == 1]

        for idx, row in historical_data.iterrows():
            driver = row["driver_id"]
            pickup = [row["pickup_latitude"], row["pickup_longitude"]]
            dropoff = [row["driver_latitude"], row["driver_longitude"]]

            driver_pickup_history.setdefault(driver, []).append(pickup)
            driver_dropoff_history.setdefault(driver, []).append(dropoff)

    # Pre-extract necessary columns
    driver_ids = df["driver_id"].values
    pickup_coords = df[["pickup_latitude", "pickup_longitude"]].values
    dropoff_coords = df[["driver_latitude", "driver_longitude"]].values

    if "is_completed" in df.columns:
        completed_flags = df["is_completed"].values
    else:
        completed_flags = None  # No completions available in test

    for i in range(len(df)):
        driver = driver_ids[i]
        pickup = pickup_coords[i]
        dropoff = dropoff_coords[i]

        # Check pickup familiarity
        if driver in driver_pickup_history:
            past_pickups = np.radians(np.array(driver_pickup_history[driver]))
            tree = BallTree(past_pickups, metric="haversine")
            query_point = np.radians([pickup])
            if len(tree.query_radius(query_point, r=radius)[0]) > 0:
                df.iat[i, df.columns.get_loc("driver_familiar_with_pickup")] = 1

        # Check dropoff familiarity
        if driver in driver_dropoff_history:
            past_dropoffs = np.radians(np.array(driver_dropoff_history[driver]))
            tree = BallTree(past_dropoffs, metric="haversine")
            query_point = np.radians([dropoff])
            if len(tree.query_radius(query_point, r=radius)[0]) > 0:
                df.iat[i, df.columns.get_loc("driver_familiar_with_dropoff")] = 1

        # Add to history only if available
        if completed_flags is not None and completed_flags[i] == 1:
            driver_pickup_history.setdefault(driver, []).append(pickup)
            driver_dropoff_history.setdefault(driver, []).append(dropoff)

    df["driver_familiarity_score"] = (
        df["driver_familiar_with_pickup"] + df["driver_familiar_with_dropoff"]
    ).astype(int)
    return df


def add_dynamic_hotspot_feature(df, historical_data=None, radius_km=0.3, window_minutes=60):
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
    df = df.sort_values("event_timestamp").copy()
    df["is_hotspot_dynamic"] = 0

    earth_radius = 6371  # in km
    radius = radius_km / earth_radius

    coords = np.radians(df[["pickup_latitude", "pickup_longitude"]].values)
    timestamps = df["event_timestamp"].values
    tree = BallTree(coords, metric='haversine')

    # If historical_data is provided, calculate dynamic hotspots based on both historical and current data
    if historical_data is not None:
        # Convert historical data timestamps to datetime
        historical_data['event_timestamp'] = pd.to_datetime(historical_data['event_timestamp'])
        historical_data = historical_data.sort_values("event_timestamp").copy()

        # Build historical hotspots (based on pickup locations)
        historical_coords = np.radians(historical_data[["pickup_latitude", "pickup_longitude"]].values)
        historical_timestamps = historical_data["event_timestamp"].values
        historical_tree = BallTree(historical_coords, metric='haversine')

        for i, (coord, ts) in enumerate(zip(coords, timestamps)):
            # Check for nearby historical hotspots within the given radius and time window
            mask_time = (historical_timestamps >= ts - np.timedelta64(window_minutes, 'm')) & (historical_timestamps < ts)
            neighbors = historical_tree.query_radius([coord], r=radius)[0]
            count_recent = np.sum(mask_time[neighbors])

            if count_recent >= 10:
                # Mark this location as a new hotspot
                df.at[df.index[i], "is_hotspot_dynamic"] = 1

    else:
        # If historical_data is None, only calculate based on current data
        for i, (coord, ts) in enumerate(zip(coords, timestamps)):
            # Check if there are enough neighbors within the time window
            mask_time = (timestamps >= ts - np.timedelta64(window_minutes, 'm')) & (timestamps < ts)
            neighbors = tree.query_radius([coord], r=radius)[0]
            count_recent = np.sum(mask_time[neighbors])

            if count_recent >= 10:
                # Mark this location as a new hotspot
                df.at[df.index[i], "is_hotspot_dynamic"] = 1

    return df



def last_5day_rolling_mean(df: pd.DataFrame, historical_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Computes the last 5-day rolling mean acceptance rate for each driver,
    using historical data if provided (for test); otherwise, uses df itself (for training).

    Parameters
    ----------
    df : pd.DataFrame
        Current batch (train or test) to apply the feature on.
    historical_data : pd.DataFrame, optional
        Past data (e.g., train_data) for rolling computation. Default is None.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with 'is_accepted_last5day_rolling_mean' feature added.
    """
    df = df.copy()

    if historical_data is None:
        historical_data = df.copy()  # In training, use df itself

    historical_data = historical_data.copy()

    # Same code as before
    historical_data['event_timestamp'] = pd.to_datetime(historical_data['event_timestamp'])
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])

    historical_data['accepted_flag'] = (historical_data['participant_status'] == 'ACCEPTED').astype(int)
    historical_data = historical_data.sort_values(['driver_id', 'event_timestamp'])
    historical_data.set_index('event_timestamp', inplace=True)

    rolling_acceptance = (
        historical_data.groupby('driver_id')['accepted_flag']
        .rolling('5D', min_periods=1)
        .mean()
        .reset_index()
        .rename(columns={'accepted_flag': 'is_accepted_last5day_rolling_mean'})
    )

    driver_last5day_mean = (
        rolling_acceptance
        .sort_values(['driver_id', 'event_timestamp'])
        .groupby('driver_id')
        .tail(1)
        .set_index('driver_id')['is_accepted_last5day_rolling_mean']
        .to_dict()
    )

    df['is_accepted_last5day_rolling_mean'] = df['driver_id'].map(driver_last5day_mean)

    global_mean = np.mean(list(driver_last5day_mean.values()))
    df['is_accepted_last5day_rolling_mean'] = df['is_accepted_last5day_rolling_mean'].fillna(global_mean)

    return df



def driver_cluster(df: pd.DataFrame, kmeans_driver: KMeans = None):
    # First, check only columns that exist
    agg_cols = [
        'driver_distance',
        'driver_familiarity_score',
        'log_acceptance_time_seconds',
        'accepted_flag',
        'historical_completed_bookings'
    ]
    available_cols = [col for col in agg_cols if col in df.columns]

    if not available_cols:
        raise ValueError("None of the required columns are available for clustering.")

    # Aggregate features
    driver_features = df.groupby('driver_id')[available_cols].mean().dropna()

    # If training (no model provided)
    if kmeans_driver is None:
        kmeans_driver = KMeans(n_clusters=5, random_state=42)
        driver_features['driver_cluster_label'] = kmeans_driver.fit_predict(driver_features)

    else:
        driver_features['driver_cluster_label'] = kmeans_driver.predict(driver_features)

    # Merge back
    df = df.merge(driver_features[['driver_cluster_label']], on='driver_id', how='left')

    return df, kmeans_driver



def customer_cluster(df, kmeans_pickup=None):
    # Cluster based on pickup location as a proxy for customer
    pickup_locations = df[['pickup_latitude', 'pickup_longitude']].dropna()

    if kmeans_pickup is None:
        # Train the KMeans model
        kmeans_pickup = KMeans(n_clusters=5, random_state=42)
        pickup_clusters = kmeans_pickup.fit_predict(pickup_locations)
    else:
        # Apply the pre-trained KMeans model to the test data
        pickup_clusters = kmeans_pickup.predict(pickup_locations)

    df.loc[pickup_locations.index, 'customer_cluster_label'] = pickup_clusters
    return df, kmeans_pickup  # Return the updated DataFrame and the KMeans model




def add_log_acceptance_time(df):
    df = df.copy()
    print(type(df['event_timestamp']))
    if not pd.api.types.is_datetime64_any_dtype(df['event_timestamp']):
        df['event_timestamp'] = df['event_timestamp'].apply(lambda x: iso_to_datetime(x))
    # Step 1: Define internal function to compute time difference for each order
    def compute_acceptance_time(each):
        each = each.copy()
        each['event_timestamp'] = pd.to_datetime(each['event_timestamp'], format='mixed', errors='coerce')
        each = each.sort_values(by='event_timestamp')

        # Get the 'CREATED' time (first CREATED event per order)
        created_ts = each.loc[each["participant_status"] == "CREATED", "event_timestamp"]

        if created_ts.empty:
            return None  # If no "CREATED" event, skip this order (this shouldn't happen if data is correct)

        created_time = created_ts.iloc[0]  # First 'CREATED' timestamp

        # Find all statuses (ACCEPTED, IGNORED, CANCELED)
        status_times = each.loc[each["participant_status"].isin(["ACCEPTED", "IGNORED", "REJECTED"])]

        if status_times.empty:
            return None  # No action taken on this order (no ACCEPTED, IGNORED, or CANCELED)

        # Now, calculate the time difference for each status separately
        results = []
        for _, row in status_times.iterrows():
            status_time = row["event_timestamp"]
            status = row["participant_status"]
            diff_seconds = (status_time - created_time).total_seconds()

            # Append the result with status and time difference
            results.append({
                "order_id": row["order_id"],
                "participant_status": status,
                "acceptance_time_seconds": diff_seconds
            })

        # Convert results into a DataFrame
        return pd.DataFrame(results)


    # Step 2: Apply per-order logic
    acceptance_time_df = (
        df.groupby("order_id")
        .apply(compute_acceptance_time)
        .dropna()
        .reset_index(drop=True)
    )

    # Step 3: Merge back with main data
    df = df.merge(
        acceptance_time_df,
        on=["order_id", "participant_status"],
        how="left"
    )

    # Step 4: Compute log-transformed version
    df["log_acceptance_time_seconds"] = np.log1p(df["acceptance_time_seconds"])

    return df

def driver_average_acceptance_time(df: pd.DataFrame, historical_data: pd.DataFrame = None) -> pd.DataFrame:
    if historical_data is not None:
        # Use historical data for calculating driver averages
        if "log_acceptance_time_seconds" not in historical_data.columns:
            historical_data = add_log_acceptance_time(historical_data)
        
        driver_avg_acceptance_time = (
            historical_data.groupby("driver_id")["log_acceptance_time_seconds"]
            .mean()
            .reset_index()
            .drop_duplicates(subset=["driver_id"])
        )
    else:
        # Otherwise, calculate from current df
        if "log_acceptance_time_seconds" not in df.columns:
            df = add_log_acceptance_time(df)
        
        driver_avg_acceptance_time = (
            df.groupby("driver_id")["log_acceptance_time_seconds"]
            .mean()
            .reset_index()
            .drop_duplicates(subset=["driver_id"])
            .rename(columns={"log_acceptance_time_seconds": "driver_avg_acceptance_time"})
        )

    # Merge back
    df = df.merge(driver_avg_acceptance_time, on="driver_id", how="left")
    if historical_data is not None:      
        df["log_acceptance_time_seconds"] = df["log_acceptance_time_seconds"].fillna(0)
    else:
        df["driver_avg_acceptance_time"] = df["driver_avg_acceptance_time"].fillna(0)
    return df



