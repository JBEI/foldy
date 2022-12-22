"""empty message

Revision ID: b55805c2389a
Revises: 40cf3787d6b4
Create Date: 2021-10-19 21:38:08.557315

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b55805c2389a'
down_revision = '40cf3787d6b4'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('roles', sa.Column('num_homooligomers', sa.Integer(), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('roles', 'num_homooligomers')
    # ### end Alembic commands ###
