"""Add source and external_id to File for Zotero

Revision ID: c1f2a3b4c5d6
Revises: ae830e7f45e0
Create Date: 2025-03-01

"""
from alembic import op
import sqlalchemy as sa


revision = 'c1f2a3b4c5d6'
down_revision = 'ae830e7f45e0'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('file', schema=None) as batch_op:
        batch_op.add_column(sa.Column('source', sa.String(20), nullable=True))
        batch_op.add_column(sa.Column('external_id', sa.String(255), nullable=True))


def downgrade():
    with op.batch_alter_table('file', schema=None) as batch_op:
        batch_op.drop_column('external_id')
        batch_op.drop_column('source')
