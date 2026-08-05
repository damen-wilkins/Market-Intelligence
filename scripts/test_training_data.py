from database.training_data_repository import TrainingDataRepository


def main():
    repository = TrainingDataRepository()

    dataframe = repository.get_training_data("SPY")

    print(dataframe.head())
    print()
    print(dataframe.info())
    print()
    print(f"Rows: {len(dataframe):,}")
    print(f"Columns: {len(dataframe.columns)}")


if __name__ == "__main__":
    main()