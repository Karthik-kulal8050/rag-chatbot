from pathlib import Path
from .parse import parse_pdf
from .chunk import chunk_pages
from .index import build_index
from .hybrid import build_bm25
from .store import save_collection, load_collection, collection_exists

def ingest_folder(folder_path, collection_id):
    """Parse+chunk+embed+index every PDF in a folder, then cache it."""
    all_chunks = []
    for pdf in Path(folder_path).glob("*.pdf"):
        all_chunks.extend(chunk_pages(parse_pdf(pdf)))

    index = build_index(all_chunks)
    bm25 = build_bm25(all_chunks)
    save_collection(collection_id, all_chunks, index, bm25)
    return all_chunks, index, bm25


def get_collection(collection_id, folder_path=None):
    """Load from cache if it exists, otherwise build and cache it."""
    if collection_exists(collection_id):
        return load_collection(collection_id)
    return ingest_folder(folder_path, collection_id)

if __name__ == "__main__":
    chunks,index, bm25 = get_collection("default", "data/default")
    print(f"Collection 'default' ready: {len(chunks)} chunks")