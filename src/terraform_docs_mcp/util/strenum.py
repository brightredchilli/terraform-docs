from enum import StrEnum


def all_values(e: type[StrEnum]) -> str:
    """
    Concatenate all members of the StrEnum and return it as a string
    """
    return ", ".join((str(p) for p in e))
