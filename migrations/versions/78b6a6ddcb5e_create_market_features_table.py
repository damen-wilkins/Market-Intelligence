"""create market_features table

Revision ID: 78b6a6ddcb5e
Revises: 10c51cf5221b
Create Date: 2026-07-22 15:15:22.497936

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '78b6a6ddcb5e'
down_revision: Union[str, Sequence[str], None] = '10c51cf5221b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "market_features",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("ticker", sa.Text(), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),

        # Trend
        sa.Column("sma_10", sa.Float()),
        sa.Column("sma_20", sa.Float()),
        sa.Column("sma_50", sa.Float()),
        sa.Column("ema_20", sa.Float()),

        # Momentum
        sa.Column("rsi_14", sa.Float()),
        sa.Column("macd", sa.Float()),
        sa.Column("macd_signal", sa.Float()),
        sa.Column("macd_histogram", sa.Float()),

        # Volatility
        sa.Column("bollinger_upper", sa.Float()),
        sa.Column("bollinger_middle", sa.Float()),
        sa.Column("bollinger_lower", sa.Float()),

        # Returns
        sa.Column("daily_return", sa.Float()),
        sa.Column("log_return", sa.Float()),

        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),

        sa.UniqueConstraint(
            "ticker",
            "trade_date",
            name="uq_market_features_ticker_trade_date",
        ),
    )

def downgrade() -> None:
    op.drop_table("market_features")
