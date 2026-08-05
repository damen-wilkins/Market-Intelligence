from app.training.dataset_validator import DatasetValidator
from app.training.label_builder import LabelBuilder
from database.training_data_repository import TrainingDataRepository


def main():
    repository = TrainingDataRepository()
    builder = LabelBuilder()
    validator = DatasetValidator()

    dataframe = repository.get_training_data("SPY")
    dataframe = builder.build_labels(dataframe)

    validator.validate(dataframe)


if __name__ == "__main__":
    main()