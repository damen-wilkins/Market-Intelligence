from app.training.stationarity_tester import StationarityTester
from database.training_data_repository import TrainingDataRepository

import matplotlib.pyplot as plt
import numpy as np
from pmdarima.arima.utils import nsdiffs
from scipy.signal import periodogram
from statsmodels.graphics.tsaplots import plot_acf


def main():
    repository = TrainingDataRepository()
    tester = StationarityTester()

    dataframe = repository.get_training_data("SPY")

    series = dataframe["log_return"].dropna()

    print("=" * 50)
    print("STATIONARITY")
    print("=" * 50)

    stationarity = tester.run(series)

    for key, value in stationarity.items():
        print(f"{key}: {value}")

    print()
    print("=" * 50)
    print("SEASONAL DIFFERENCING")
    print("=" * 50)

    ocsb = nsdiffs(
        series,
        test="ocsb",
        m=5
    )

    ch = nsdiffs(
        series,
        test="ch",
        m=5
    )

    print(f"OCSB Recommendation : D = {ocsb}")
    print(f"Canova-Hansen       : D = {ch}")

    print()
    print("=" * 50)
    print("PERIODOGRAM")
    print("=" * 50)

    frequencies, power = periodogram(series)

    peaks = np.argsort(power)[-10:][::-1]

    for peak in peaks:
        if frequencies[peak] == 0:
            continue

        period = 1 / frequencies[peak]

        print(
            f"Period: {period:8.2f}   "
            f"Power: {power[peak]:.6f}"
        )

    plt.figure(figsize=(12, 5))
    plt.plot(frequencies, power)
    plt.title("Periodogram")
    plt.xlabel("Frequency")
    plt.ylabel("Power")
    plt.tight_layout()
    plt.show()

    plot_acf(
        series,
        lags=100
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()