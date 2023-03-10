"""empty message

Revision ID: 291e8e8dbab1
Revises: b55805c2389a
Create Date: 2021-10-20 20:06:53.929583

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "291e8e8dbab1"
down_revision = "b55805c2389a"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column("roles", sa.Column("tagstring", sa.Text(), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("roles", "tagstring")
    # ### end Alembic commands ###
