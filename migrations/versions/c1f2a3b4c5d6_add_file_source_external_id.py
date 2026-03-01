"""Add source and external_id to File for Zotero

Revision ID: c1f2a3b4c5d6
Revises: ae830e7f45e0
Create Date: 2025-03-01

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = 'c1f2a3b4c5d6'
down_revision = 'ae830e7f45e0'
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    inspector = inspect(conn)
    file_columns = [c["name"] for c in inspector.get_columns("file")]
    with op.batch_alter_table("file", schema=None) as batch_op:
        if "source" not in file_columns:
            batch_op.add_column(sa.Column("source", sa.String(20), nullable=True))
        if "external_id" not in file_columns:
            batch_op.add_column(sa.Column("external_id", sa.String(255), nullable=True))


def downgrade():
    conn = op.get_bind()
    inspector = inspect(conn)
    file_columns = [c["name"] for c in inspector.get_columns("file")]
    with op.batch_alter_table("file", schema=None) as batch_op:
        if "external_id" in file_columns:
            batch_op.drop_column("external_id")
        if "source" in file_columns:
            batch_op.drop_column("source")
