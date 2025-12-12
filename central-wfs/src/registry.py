from dataclasses import dataclass
from typing import Dict, Tuple, Literal
from datetime import ZoneInfo
from typing import Type, Dict
from src.base import BaseFormat, BaseSite

# SITE_REGISTRY = {}

@dataclass(frozen=True)
class SiteInfo:
    site: str
    root_path: str
    metadata_path: str
    mapping_path: str
    bedside_system: Tuple[Literal["DRAGER", "DWC", "PHILIPS", "GE", "NIHON KOHDEN"], ...]
    backend_system: Tuple[Literal["CAPSULE", "BEDMASTER", "SICKBAY", "DWC"], ...]
    file_formats: Tuple[Literal["CSV","H5","PARQUET","FEATHER","WFDB","DICOM"],...]
    is_wfdb: bool
    telemetry: bool
    eeg: bool
    timezone: ZoneInfo
    time_base: Literal["STREAM-BASED", "PAIR-BASED"]

class Registry:
    """
    Class methods for defining and loading site settings prior to
    executing ingestion, correction, and conversion pipeline.
    """

    @classmethod
    def register_site(cls, site_name: str, converter_config: SiteInfo):
        cls.site_config = converter_config

    @classmethod
    def get_site_config(cls, site_name: str):
        try:
            return cls.site_config
        except KeyError:
            raise ValueError(f"No loader registered for site '{site_name}'")