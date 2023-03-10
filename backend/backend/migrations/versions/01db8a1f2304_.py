"""empty message

Revision ID: 01db8a1f2304
Revises: 95d0e6ec9b80
Create Date: 2021-09-29 00:36:59.544288

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "01db8a1f2304"
down_revision = "95d0e6ec9b80"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column("roles", sa.Column("msa_size_gb", sa.Float(), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("roles", "msa_size_gb")
    # ### end Alembic commands ###
