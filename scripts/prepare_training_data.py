from app.training.data_splitter import DataSplitter
from app.training.dataset_validator import DatasetValidator
from app.training.label_builder import LabelBuilder
from database.training_data_repository import TrainingDataRepository


def main():
    repository = TrainingDataRepository()
    builder = LabelBuilder()
    splitter = DataSplitter()
    validator = DatasetValidator()

    dataframe = repository.get_training_data("SPY")
    dataframe = builder.build_labels(dataframe)

    train, validation, test = splitter.split(dataframe)

    print("\nTraining Set")
    validator.validate(train)

    print("\nValidation Set")
    validator.validate(validation)

    print("\nTest Set")
    validator.validate(test)

    print("\nDataset Split")
    print(f"Training:   {len(train):,}")
    print(f"Validation: {len(validation):,}")
    print(f"Testing:    {len(test):,}")


if __name__ == "__main__":
    main()