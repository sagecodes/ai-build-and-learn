async def ingest(data_dir: str) -> None:
    """Ingest all PDFs via cognee.add() + cognee.cognify() into local LanceDB storage."""
    raise NotImplementedError


if __name__ == "__main__":
    import asyncio
    from config import DATA_DIR
    asyncio.run(ingest(str(DATA_DIR)))
