"""add_macro_feature_tables

Revision ID: 0298e8827407
Revises: 78b6a6ddcb5e
Create Date: 2026-07-25 19:09:24.202629

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0298e8827407"
down_revision: Union[str, Sequence[str], None] = "78b6a6ddcb5e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "feature_catalog",
        sa.Column("feature_id", sa.Integer(), primary_key=True),
        sa.Column("feature_name", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("provider_symbol", sa.Text(), nullable=False, unique=True),
        sa.Column(
            "active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
    )

    op.create_table(
        "macro_features",
        sa.Column("feature_id", sa.Integer(), nullable=False),
        sa.Column("observation_date", sa.Date(), nullable=False),
        sa.Column("value", sa.Numeric(), nullable=False),
        sa.ForeignKeyConstraint(
            ["feature_id"],
            ["feature_catalog.feature_id"],
        ),
        sa.PrimaryKeyConstraint(
            "feature_id",
            "observation_date",
        ),
    )


def downgrade() -> None:
    op.drop_table("macro_features")
    op.drop_table("feature_catalog")