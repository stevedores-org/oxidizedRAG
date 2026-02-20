"""Utility functions for the sample application."""


def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    return text.lower().strip().replace(" ", "-")


def truncate(text: str, max_length: int = 100) -> str:
    """Truncate text to max_length, adding ellipsis if needed."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def chunk_list(items: list, chunk_size: int) -> list:
    """Split a list into chunks of the given size."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
