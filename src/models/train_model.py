from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.models.classifier import SklearnClassifier
from src.utils.config import load_config
from src.utils.guardrails import validate_evaluation_metrics
from src.utils.store import AssignmentStore


@validate_evaluation_metrics
def main():
    store = AssignmentStore()
    config = load_config()

    df = store.get_processed("transformed_dataset.csv")
    df_train, df_test = train_test_split(df, test_size=config["test_size"])
    scaler= StandardScaler()
    df_train.loc[:, config["features"]] = scaler.fit_transform(df_train[config["features"]])
    df_test.loc[:, config["features"]] = scaler.transform(df_test[config["features"]])
    df_train.loc[:, config["target"]] = df_train[config["target"]].astype(int)
    df_test.loc[:, config["target"]] = df_test[config["target"]].astype(int)
    
    rf_estimator = RandomForestClassifier(**config["random_forest"])
    model = SklearnClassifier(rf_estimator, config["features"], config["target"])
    model.train(df_train)
    feature_importances = model.clf.feature_importances_
    print(f"Feature Importances: {feature_importances}")
    metrics = model.evaluate(df_test)

    store.put_model("saved_model.pkl", model)
    store.put_metrics("metrics.json", metrics)


if __name__ == "__main__":
    main()
