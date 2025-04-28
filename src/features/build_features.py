import pandas as pd
from src.utils.store import AssignmentStore

store = AssignmentStore()
from src.features.transformations import (
    driver_distance_to_pickup,
    driver_historical_completed_bookings,
    hour_of_day,
    add_driver_familiarity_features,
    add_dynamic_hotspot_feature,
    driver_cluster,
    customer_cluster,
    drop_near_next_pickup,
    last_5day_rolling_mean,
    driver_average_acceptance_time
)


def main():
    store = AssignmentStore()
    
    dataset = store.get_processed("dataset.csv")
    dataset = apply_feature_engineering(dataset)
    dataset = dataset[dataset['participant_status'] != "CREATED"]
    store.put_processed("transformed_dataset.csv", dataset)


def apply_feature_engineering(df: pd.DataFrame, historical_data: pd.DataFrame = None,c_cluster=None,d_cluster=None) -> pd.DataFrame:

    df = driver_distance_to_pickup(df)
    print("After driver_distance_to_pickup")
    
    df= hour_of_day(df)
    print("After hour_of_day")

    df = driver_historical_completed_bookings(df, historical_data)
    print("After driver_historical_completed_bookings")


    df = add_driver_familiarity_features(df,historical_data)
    print("After add_driver_familiarity_features")
    
    df = add_dynamic_hotspot_feature(df)
    print("After add_dynamic_hotspot_feature")
    
    
    df,pickup_cluster = customer_cluster(df,c_cluster)
    store.put_model("pickup_cluster.pkl", pickup_cluster)
    print("After customer_cluster")
    

    df = drop_near_next_pickup(df,historical_data)
    print("After drop_near_next_pickup")
    
    df = last_5day_rolling_mean(df,historical_data)
    print("After last_5day_rolling_mean")
    
    df,driver_location=driver_cluster(df,d_cluster)
    store.put_model("driver_cluster.pkl", driver_location)
    print("After driver_cluster")
    # print(df)
    df=driver_average_acceptance_time(df,historical_data)
    print("After driver_average_acceptance_time")
    return df



if __name__ == "__main__":
    main()
