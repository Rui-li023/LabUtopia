"""Chemical substance data classes."""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class SubstanceProperty:
    """Intrinsic properties of a chemical substance (immutable across operations)."""

    name: str  # e.g. "hydrochloric_acid", "water"
    formula: str  # e.g. "HCl", "H2O"
    molecular_weight: float  # g/mol
    state: str  # "solid" | "liquid" | "gas"
    density: float  # g/mL at standard conditions
    boiling_point: float  # °C
    melting_point: float  # °C
    color: Tuple[float, ...] = field(default=(1.0, 1.0, 1.0, 1.0))  # RGBA

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "formula": self.formula,
            "molecular_weight": self.molecular_weight,
            "state": self.state,
            "density": self.density,
            "boiling_point": self.boiling_point,
            "melting_point": self.melting_point,
            "color": list(self.color),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubstanceProperty":
        """Deserialize from dictionary."""
        data = dict(data)
        data["color"] = tuple(data["color"])
        return cls(**data)


@dataclass
class SubstanceInstance:
    """An instance of a substance inside a container, with quantity and state."""

    property: SubstanceProperty
    volume_ml: float  # current volume in mL
    concentration_mol_l: float = 0.0  # mol/L (0 for pure substances)
    temperature_c: float = 25.0  # current temperature in °C

    @property
    def name(self) -> str:
        """Shortcut to the substance name."""
        return self.property.name

    @property
    def mass_g(self) -> float:
        """Current mass in grams, computed from volume and density."""
        return self.volume_ml * self.property.density

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "property": self.property.to_dict(),
            "volume_ml": self.volume_ml,
            "concentration_mol_l": self.concentration_mol_l,
            "temperature_c": self.temperature_c,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubstanceInstance":
        """Deserialize from dictionary."""
        return cls(
            property=SubstanceProperty.from_dict(data["property"]),
            volume_ml=data["volume_ml"],
            concentration_mol_l=data["concentration_mol_l"],
            temperature_c=data["temperature_c"],
        )
