import pandas as pd


class DatasetValidator:
    def validate(self, dataframe: pd.DataFrame) -> None:
        print("\n========== DATASET VALIDATION ==========")

        duplicate_dates = dataframe["trade_date"].duplicated().sum()

        if duplicate_dates == 0:
            print("✓ No duplicate trade dates")
        else:
            print(f"✗ Duplicate trade dates: {duplicate_dates}")

        if dataframe["trade_date"].is_monotonic_increasing:
            print("✓ Trade dates are in chronological order")
        else:
            print("✗ Trade dates are not in chronological order")

        print("\nMissing Values")
        missing = dataframe.isnull().sum()
        missing = missing[missing > 0]

        if missing.empty:
            print("✓ No missing values")
        else:
            print(missing)

        print("\nLabel Validation")

        for horizon in ["1d", "1w", "1m", "1y"]:
            column = f"label_{horizon}"
            missing_labels = dataframe[column].isnull().sum()

            if missing_labels == 0:
                print(f"✓ {column}: Complete")
            else:
                print(f"• {column}: {missing_labels} missing")

        print("\n========================================")