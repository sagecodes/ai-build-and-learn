def ingest(data_dir: str) -> None:
    """Embed all PDFs from data_dir and store in Supabase pgvector."""
    raise NotImplementedError


if __name__ == "__main__":
    from config import DATA_DIR
    ingest(str(DATA_DIR))
