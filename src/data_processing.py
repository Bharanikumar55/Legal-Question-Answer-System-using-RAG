"""
Data Processing Module - PDF to Chunks Pipeline
Handles: PDF text extraction, cleaning, and chunking
"""

import pdfplumber
import re
import json
import os
from pathlib import Path


def extract_text(pdf_path):
    """Extract text from PDF file"""
    full_text = ""
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                full_text += text + " "
    
    return full_text


def clean_text(text):
    """Clean and normalize extracted text"""
    text = re.sub(r'\n+', ' ', text)   # remove newlines
    text = re.sub(r'\s+', ' ', text)   # remove extra spaces
    return text.strip()


def chunk_text(text, chunk_size=500, overlap=100):
    """Chunk text into overlapping segments"""
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        chunk = text[start:start + chunk_size]
        chunks.append({
            "chunk_id": chunk_id,
            "content": chunk
        })
        start += chunk_size - overlap
        chunk_id += 1
    
    return chunks


def process_pdf(pdf_path, chunk_size=500, overlap=100):
    """Complete pipeline: Extract → Clean → Chunk"""
    raw_text = extract_text(pdf_path)
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text, chunk_size, overlap)
    return chunks


def save_chunks_json(chunks, output_file):
    """Save chunks to JSON file"""
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(chunks, f, indent=4)


def process_all_pdfs(data_dir="../data", output_dir="../data/processed"):
    """Process all PDFs in a directory"""
    os.makedirs(output_dir, exist_ok=True)
    data_dir = Path(data_dir)
    
    results = {
        "total_files": 0,
        "successful": 0,
        "failed": 0,
        "files": []
    }
    
    pdf_files = list(data_dir.glob("*.pdf"))
    results["total_files"] = len(pdf_files)
    
    for pdf_file in pdf_files:
        try:
            print(f"Processing: {pdf_file.name}")
            chunks = process_pdf(str(pdf_file))
            output_file = os.path.join(output_dir, f"{pdf_file.stem}_chunks.json")
            save_chunks_json(chunks, output_file)
            
            results["successful"] += 1
            results["files"].append({
                "filename": pdf_file.name,
                "chunks": len(chunks),
                "output": output_file
            })
            print(f"✓ {len(chunks)} chunks generated")
        except Exception as e:
            results["failed"] += 1
            results["files"].append({
                "filename": pdf_file.name,
                "error": str(e)
            })
            print(f"✗ Error: {str(e)}")
    
    return results


# ===== TEMPORARY TEST CODE =====
if __name__ == "__main__":
    print("Testing data_processing module...\n")
    
    # Test 1: Check if data folder exists
    print("Test 1: Checking data directory...")
    # Get the project root (parent of src directory)
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    data_dir = project_root / "data"
    if data_dir.exists():
        pdfs = list(data_dir.glob("*.pdf"))
        print(f"✓ Found {len(pdfs)} PDF files: {[p.name for p in pdfs]}\n")
    else:
        print(f"✗ Data directory not found at {data_dir}\n")
    
    # Test 2: Extract and clean sample text
    print("Test 2: Testing text cleaning...")
    sample_text = "Line 1\n\n\nLine 2\n\n  Line 3   "
    cleaned = clean_text(sample_text)
    print(f"Original: {repr(sample_text)}")
    print(f"Cleaned:  {repr(cleaned)}")
    print(f"✓ Cleaning works\n")
    
    # Test 3: Chunk text
    print("Test 3: Testing text chunking...")
    long_text = "This is a sample text. " * 50  # ~1150 chars
    chunks = chunk_text(long_text, chunk_size=500, overlap=100)
    print(f"Text length: {len(long_text)} chars")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Chunk 0 size: {len(chunks[0]['content'])} chars")
    if len(chunks) > 1:
        print(f"Chunk 1 size: {len(chunks[1]['content'])} chars")
        # Check overlap
        overlap_text = chunks[0]['content'][-100:]
        chunk1_start = chunks[1]['content'][:100]
        if overlap_text == chunk1_start:
            print(f"✓ Overlap preserved correctly\n")
        else:
            print(f"⚠ Overlap not matching as expected\n")
    
    # Test 4: Process a real PDF if available
    print("Test 4: Testing PDF processing...")
    pdf_files = list((project_root / "data").glob("*.pdf"))
    if pdf_files:
        try:
            pdf_path = str(pdf_files[0])
            chunks = process_pdf(pdf_path)
            print(f"✓ Processed {pdf_files[0].name}")
            print(f"  Total chunks: {len(chunks)}")
            print(f"  Sample chunk content (first 100 chars): {chunks[0]['content'][:100]}...\n")
        except Exception as e:
            print(f"✗ Error processing PDF: {e}\n")
    else:
        print("✗ No PDF files found\n")
    
    # Test 5: Batch process all PDFs
    print("Test 5: Batch processing all PDFs...")
    try:
        data_dir_str = str(project_root / "data")
        output_dir_str = str(project_root / "data" / "processed")
        results = process_all_pdfs(data_dir=data_dir_str, output_dir=output_dir_str)
        print(f"\nBatch Results:")
        print(f"  Total files: {results['total_files']}")
        print(f"  Successful: {results['successful']}")
        print(f"  Failed: {results['failed']}")
        for file_info in results['files']:
            if 'chunks' in file_info:
                print(f"  ✓ {file_info['filename']}: {file_info['chunks']} chunks")
            else:
                print(f"  ✗ {file_info['filename']}: {file_info['error']}")
    except Exception as e:
        print(f"✗ Batch processing error: {e}")
    
    print("\n" + "="*50)
    print("Testing complete!")
    print("="*50)
