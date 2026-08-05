from database.training_data_repository import TrainingDataRepository
from app.training.label_builder import LabelBuilder
from app.training.data_splitter import DataSplitter


def audit(name, dataframe):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Rows: {len(dataframe):,}")
    print(f"Columns: {len(dataframe.columns)}")
    print(f"Duplicate dates: {dataframe['trade_date'].duplicated().sum()}")
    print(f"Missing values: {int(dataframe.isna().sum().sum())}")
    print(f"Date range: {dataframe['trade_date'].min()} -> {dataframe['trade_date'].max()}")


def main():
    repository = TrainingDataRepository()
    builder = LabelBuilder()
    splitter = DataSplitter()

    dataframe = repository.get_training_data("SPY")
    dataframe = builder.build_labels(dataframe)

    train, validation, test = splitter.split(dataframe)

    audit("Training", train)
    audit("Validation", validation)
    audit("Test", test)


if __name__ == "__main__":
    main()