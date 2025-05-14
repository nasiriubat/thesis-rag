"""add file identifiers

Revision ID: add_file_identifiers
Revises: previous_revision
Create Date: 2024-03-14 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_file_identifiers'
down_revision = 'previous_revision'  # replace with your previous revision
branch_labels = None
depends_on = None

def upgrade():
    # Add new columns to File table
    op.add_column('file', sa.Column('file_identifier', sa.String(100), nullable=True))
    op.add_column('file', sa.Column('original_filename', sa.String(255), nullable=True))
    
    # Create index on file_identifier for faster lookups
    op.create_index(op.f('ix_file_file_identifier'), 'file', ['file_identifier'], unique=True)

def downgrade():
    # Remove index
    op.drop_index(op.f('ix_file_file_identifier'), table_name='file')
    
    # Remove columns
    op.drop_column('file', 'original_filename')
    op.drop_column('file', 'file_identifier') 