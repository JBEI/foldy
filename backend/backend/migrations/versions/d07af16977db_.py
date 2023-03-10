"""empty message

Revision ID: d07af16977db
Revises: 50906750b751
Create Date: 2021-09-23 21:45:59.338315

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "d07af16977db"
down_revision = "50906750b751"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column("roles", sa.Column("rq_job_id", sa.String(length=80), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("roles", "rq_job_id")
    # ### end Alembic commands ###
