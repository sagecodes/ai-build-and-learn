def ingest(data_dir: str) -> None:
    """Extract entities and relationships from PDFs and store in Neo4j AuraDB."""
    raise NotImplementedError


if __name__ == "__main__":
    from config import DATA_DIR
    ingest(str(DATA_DIR))
