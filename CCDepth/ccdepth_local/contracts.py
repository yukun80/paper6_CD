from dataclasses import dataclass, field
from typing import Any
import numpy as np

@dataclass
class GridSpec:
    crs: str
    transform: tuple
    height: int
    width: int
    row_offset: int = 0
    col_offset: int = 0
    profile: dict = field(default_factory=dict)
    @property
    def shape(self): return (self.height, self.width)

@dataclass
class RasterInputs:
    dem: np.ndarray
    flood: np.ndarray
    mask_valid: np.ndarray
    dem_valid: np.ndarray
    grid: GridSpec
    hashes: dict = field(default_factory=dict)

@dataclass
class DomainData:
    observed: np.ndarray
    support: np.ndarray
    dry: np.ndarray
    hard: np.ndarray
    soft: np.ndarray
    boundary: np.ndarray
    component_id: np.ndarray
    component_table: Any

@dataclass
class TopologyData:
    rows: np.ndarray
    cols: np.ndarray
    neighbors: np.ndarray
    weights: np.ndarray
    degree: np.ndarray
    color: np.ndarray
    shape: tuple

@dataclass
class BoundaryData:
    lower: np.ndarray
    upper: np.ndarray
    mid: np.ndarray
    beta0: np.ndarray | None
    beta: np.ndarray
    wet_count: np.ndarray | None
    dry_count: np.ndarray | None
    diagnostic: np.ndarray | None
    eligible: bool
    initial_S: np.ndarray

@dataclass
class ComponentState:
    status: int = 1
    attempt: int = 0
    sweeps: int = 0
    stable: int = 0
    prev_primary: float = 0
    prev_total: float = 0
    base_primary: float = 0
    base_mid: float = 0

@dataclass
class Prepared:
    topology: TopologyData
    dem: np.ndarray
    hard: np.ndarray
    boundary: BoundaryData
    component_id: int = 1

@dataclass
class SolverResult:
    S: np.ndarray
    baseS: np.ndarray
    state: ComponentState
    history: list
    audit: dict
    total_sweeps: int = 0
    reason: str = ""
