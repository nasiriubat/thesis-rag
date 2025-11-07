"""
Script to process all existing files in the database and extract knowledge graph triples.
This can be run on an existing production database to add KAG support to existing files.

Usage:
    python process_existing_files_for_kag.py              # Process all files
    python process_existing_files_for_kag.py --force      # Reprocess files that already have KG
    python process_existing_files_for_kag.py --dry-run    # Show what would be processed without doing it
"""
from app import create_app
from models import db, File, Settings, KnowledgeGraph
from knowledge_graph import process_file_to_graph, delete_document_graph
from openai import OpenAI
from sqlalchemy import inspect
import sys
import argparse

def main():
    """Process all existing files for knowledge graph extraction."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process existing files for knowledge graph extraction')
    parser.add_argument('--force', action='store_true', 
                       help='Reprocess files that already have knowledge graph entries')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without actually processing')
    args = parser.parse_args()
    
    app = create_app()
    
    with app.app_context():
        # Check if knowledge_graph table exists
        inspector = inspect(db.engine)
        existing_tables = inspector.get_table_names()
        
        if 'knowledge_graph' not in existing_tables:
            print("❌ Error: knowledge_graph table does not exist!")
            print("Please run 'python init_db.py' first to create the table.")
            sys.exit(1)
        
        # Check if OpenAI key is configured
        openai_key = Settings.query.filter_by(key="openai_key").first()
        if not openai_key or not openai_key.value:
            print("❌ Error: OpenAI API key is not configured!")
            print("Please set 'openai_key' in Settings (Admin → Settings).")
            sys.exit(1)
        
        # Initialize OpenAI client
        try:
            client = OpenAI(api_key=openai_key.value)
        except Exception as e:
            print(f"❌ Error initializing OpenAI client: {e}")
            sys.exit(1)
        
        # Get all files
        all_files = File.query.all()
        total_files = len(all_files)
        
        if total_files == 0:
            print("ℹ️  No files found in database.")
            return
        
        print(f"📊 Found {total_files} files to process")
        print(f"🔑 Using OpenAI key: {openai_key.value[:10]}...")
        print("-" * 60)
        
        # Check which files already have KG entries
        files_with_kg = db.session.query(KnowledgeGraph.source_id).distinct().all()
        files_with_kg_ids = set([f[0] for f in files_with_kg if f[0] is not None])
        
        # Filter files to process
        if args.force:
            files_to_process = all_files  # Process all files
            print("⚠️  --force flag: Will reprocess files that already have KG entries")
        else:
            files_to_process = [f for f in all_files if f.id not in files_with_kg_ids]
        
        if len(files_to_process) == 0:
            print("✅ All files already have knowledge graph triples!")
            print(f"   Total files: {total_files}")
            print(f"   Files with KG: {len(files_with_kg_ids)}")
            return
        
        print(f"📝 Files to process: {len(files_to_process)}")
        if not args.force:
            print(f"✅ Files already processed: {len(files_with_kg_ids)}")
        print("-" * 60)
        
        if args.dry_run:
            print("\n🔍 DRY RUN MODE - No files will be processed")
            print("\nFiles that would be processed:")
            for idx, file in enumerate(files_to_process[:10], 1):  # Show first 10
                filename = file.original_filename or f"file_{file.id}"
                has_kg = file.id in files_with_kg_ids
                print(f"  {idx}. {filename} (ID: {file.id}, Has KG: {has_kg}, Text length: {len(file.text) if file.text else 0})")
            if len(files_to_process) > 10:
                print(f"  ... and {len(files_to_process) - 10} more files")
            print(f"\nTotal: {len(files_to_process)} files would be processed")
            return
        
        # Ask for confirmation
        response = input(f"\nDo you want to process {len(files_to_process)} files? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("❌ Cancelled by user.")
            return
        
        print("\n🚀 Starting processing...\n")
        
        # Process files
        processed_count = 0
        error_count = 0
        skipped_count = 0
        
        for idx, file in enumerate(files_to_process, 1):
            file_id = file.id
            filename = file.original_filename or f"file_{file_id}"
            
            # Skip files with empty text
            if not file.text or not file.text.strip():
                print(f"[{idx}/{len(files_to_process)}] ⏭️  Skipped: {filename} (empty text)")
                skipped_count += 1
                continue
            
            # Check text length (skip very short files)
            if len(file.text.strip()) < 50:
                print(f"[{idx}/{len(files_to_process)}] ⏭️  Skipped: {filename} (text too short: {len(file.text.strip())} chars)")
                skipped_count += 1
                continue
            
            try:
                print(f"[{idx}/{len(files_to_process)}] 🔄 Processing: {filename} ({len(file.text)} chars)...", end=" ")
                
                # If force mode and file already has KG, delete existing entries first
                if args.force and file_id in files_with_kg_ids:
                    delete_document_graph(file_id)
                
                # Process file to knowledge graph
                success = process_file_to_graph(file_id, file.text, client)
                
                if success:
                    # Count triples for this file
                    triple_count = KnowledgeGraph.query.filter_by(source_id=file_id).count()
                    print(f"✅ Done ({triple_count} triples)")
                    processed_count += 1
                else:
                    print("⚠️  No triples extracted")
                    error_count += 1
                    
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                error_count += 1
                continue
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Processing Summary:")
        print("=" * 60)
        print(f"✅ Successfully processed: {processed_count} files")
        print(f"❌ Errors: {error_count} files")
        print(f"⏭️  Skipped: {skipped_count} files")
        print(f"📁 Total files: {total_files}")
        
        # Get final stats
        total_triples = KnowledgeGraph.query.count()
        files_with_kg = db.session.query(KnowledgeGraph.source_id).distinct().count()
        
        print("\n📈 Knowledge Graph Statistics:")
        print(f"   Total triples: {total_triples}")
        print(f"   Files with KG: {files_with_kg}")
        print(f"   Unique subjects: {db.session.query(KnowledgeGraph.subject).distinct().count()}")
        print(f"   Unique relations: {db.session.query(KnowledgeGraph.relation).distinct().count()}")
        
        if processed_count > 0:
            print("\n✅ Processing complete! You can now use KAG retrieval.")
        else:
            print("\n⚠️  No files were processed. Check the errors above.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Partial processing may have occurred.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

