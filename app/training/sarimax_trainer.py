from app.training.sarimax_parameter_selector import SarimaxParameterSelector

class SarimaxTrainer:
    def __init__(self):
        self.selector = SarimaxParameterSelector()

    def train(
        self,
        endog,
        exog=None,
    ):
        selection = self.selector.select(
            endog=endog,
            exog=exog,
        )

        return {
            "model": selection["model"],
            "order": selection["order"],
            "seasonal_order": selection["seasonal_order"],
            "aicc": selection["aicc"],
        }