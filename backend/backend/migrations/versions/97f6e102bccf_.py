"""empty message

Revision ID: 97f6e102bccf
Revises: 6c4d9d4a8de3
Create Date: 2023-07-15 00:22:20.205512

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '97f6e102bccf'
down_revision = '6c4d9d4a8de3'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('invokation', sa.Column('starttime', sa.DateTime(), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('invokation', 'starttime')
    # ### end Alembic commands ###
