import asyncio
from typing import Any


def cache_result(func):
    store = {}

    async def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key in store:
            return store[key]
        value = await func(*args, **kwargs)
        store[key] = value
        return value

    return wrapper


class DataProcessor:
    def __init__(self, source: str):
        self.source = source

    def normalize_name(self, name: str) -> str:
        return "_".join(part for part in name.lower().split(" ") if part)

    @cache_result
    async def fetch_records(self) -> list[dict[str, Any]]:
        await asyncio.sleep(0)
        return [{"id": 1, "name": "Alice"}]
