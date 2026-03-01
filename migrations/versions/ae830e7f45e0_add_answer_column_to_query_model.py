"""Add answer column to Query model

Revision ID: ae830e7f45e0
Revises: 0e051dd41b8d
Create Date: 2025-05-14 13:14:26.767976

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = 'ae830e7f45e0'
down_revision = '0e051dd41b8d'
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    inspector = inspect(conn)
    query_columns = [c["name"] for c in inspector.get_columns("query")]
    if "answer" not in query_columns:
        with op.batch_alter_table("query", schema=None) as batch_op:
            batch_op.add_column(sa.Column("answer", sa.Text(), nullable=True))


def downgrade():
    conn = op.get_bind()
    inspector = inspect(conn)
    query_columns = [c["name"] for c in inspector.get_columns("query")]
    if "answer" in query_columns:
        with op.batch_alter_table("query", schema=None) as batch_op:
            batch_op.drop_column("answer")
