from database.training_data_repository import TrainingDataRepository
from app.training.label_builder import LabelBuilder

def main():
    repository = TrainingDataRepository()
    builder = LabelBuilder()

    dataframe = repository.get_training_data("SPY")
    dataframe = builder.build_labels(dataframe)

    for horizon in ["1d", "1w", "1m", "1y"]:
        print(f"\n{horizon.upper()} Labels")
        print(dataframe[f"label_{horizon}"].value_counts(dropna=False))

if __name__ == "__main__":
    main()