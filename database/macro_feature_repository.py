import pandas as pd
from sqlalchemy import create_engine, text
from database.connection import get_connection_string

class MacroFeatureRepository:
    def __init__(self):
        self.engine = create_engine(get_connection_string())

    def get_active_features(self):
        query = """
            SELECT
                feature_id,
                feature_name,
                provider,
                provider_symbol
            FROM feature_catalog
            WHERE active = TRUE
            ORDER BY feature_id;
        """

        with self.engine.begin() as connection:
            return connection.execute(text(query)).mappings().all()
    def get_training_features(self):
        query = """
            SELECT
                feature_id,
                feature_name,
                provider,
                provider_symbol
            FROM feature_catalog
            WHERE active = TRUE
              AND training_feature = TRUE
            ORDER BY feature_id;
        """
        with self.engine.begin() as connection:
            return connection.execute(text(query)).mappings().all()
    def get_training_feature_data(self) -> pd.DataFrame:
        query = """
            SELECT
                mf.observation_date,
                fc.feature_name,
                mf.value
            FROM macro_features mf
            JOIN feature_catalog fc
                ON mf.feature_id = fc.feature_id
            WHERE fc.active = TRUE
              AND fc.training_feature = TRUE
            ORDER BY mf.observation_date, fc.feature_name;
        """
        dataframe = pd.read_sql(query, self.engine)
        if dataframe.empty:
            return dataframe
        return (
            dataframe
            .pivot(
                index="observation_date",
                columns="feature_name",
                values="value"
            )
            .reset_index()
            .sort_values("observation_date")
        )
    def get_latest_observation_date(self, feature_id: int):
        query = """
            SELECT
                MAX(observation_date)
            FROM macro_features
            WHERE feature_id = :feature_id;
        """
        with self.engine.begin() as connection:
            return connection.execute(
                text(query),
                {"feature_id": feature_id},
            ).scalar()
    def insert_feature_catalog(self, features):
        query = """
            INSERT INTO feature_catalog (
                feature_name,
                provider,
                provider_symbol,
                active,
                training_feature
            )
            VALUES (
                :feature_name,
                :provider,
                :provider_symbol,
                :active,
                :training_feature
            )
            ON CONFLICT (provider_symbol)
            DO UPDATE SET
                feature_name = EXCLUDED.feature_name,
                provider = EXCLUDED.provider,
                active = EXCLUDED.active,
                training_feature = EXCLUDED.training_feature;
        """
        with self.engine.begin() as connection:
            connection.execute(text(query), features)
    def upsert_macro_features(self, feature_id: int, dataframe: pd.DataFrame):
        query = """
            INSERT INTO macro_features (
                feature_id,
                observation_date,
                value
            )
            VALUES (
                :feature_id,
                :observation_date,
                :value
            )
            ON CONFLICT (feature_id, observation_date)
            DO UPDATE SET
                value = EXCLUDED.value;
        """
        rows = []
        for _, row in dataframe.iterrows():
            if pd.isna(row["value"]):
                continue
            rows.append(
                {
                    "feature_id": feature_id,
                    "observation_date": row["observation_date"],
                    "value": float(row["value"]),
                }
            )
        if not rows:
            return
        with self.engine.begin() as connection:
            connection.execute(text(query), rows)