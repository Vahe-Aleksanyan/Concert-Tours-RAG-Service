from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("concert_ingestion_service")
except PackageNotFoundError:
    __version__ = "0.1.0"
