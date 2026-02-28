"""Chemical container bound to a USD prim path."""

from typing import Any, Dict, List, Optional

from loguru import logger

from chemistry.substance import SubstanceInstance, SubstanceProperty


class Container:
    """A chemical container in the simulation scene.

    Each container is bound to a USD prim path and tracks the substances
    it holds along with their volumes, concentrations, and temperatures.
    """

    def __init__(
        self,
        usd_path: str,
        container_type: str,
        capacity_ml: float,
    ) -> None:
        self.usd_path = usd_path  # e.g. "/World/beaker_01"
        self.container_type = container_type  # "beaker" | "flask" | "test_tube" | ...
        self.capacity_ml = capacity_ml
        self._substances: List[SubstanceInstance] = []

    # ------------------------------------------------------------------
    # Query properties
    # ------------------------------------------------------------------

    @property
    def total_volume_ml(self) -> float:
        """Total volume of all substances in this container."""
        return sum(s.volume_ml for s in self._substances)

    @property
    def is_empty(self) -> bool:
        """Whether the container holds no substances."""
        return len(self._substances) == 0

    @property
    def remaining_capacity_ml(self) -> float:
        """Remaining capacity in mL."""
        return self.capacity_ml - self.total_volume_ml

    @property
    def substances(self) -> List[SubstanceInstance]:
        """Read-only view of current substances."""
        return list(self._substances)

    def get_substance(self, name: str) -> Optional[SubstanceInstance]:
        """Find a substance by name, or return None."""
        for s in self._substances:
            if s.name == name:
                return s
        return None

    # ------------------------------------------------------------------
    # Mutation methods
    # ------------------------------------------------------------------

    def add_substance(self, substance: SubstanceInstance) -> None:
        """Add a substance instance to this container.

        If the container already holds the same substance, merge volumes
        using a weighted average for concentration and temperature.
        """
        if substance.volume_ml <= 0:
            logger.warning(
                "Ignoring add_substance with non-positive volume: {}",
                substance.volume_ml,
            )
            return

        if substance.volume_ml > self.remaining_capacity_ml + 1e-6:
            logger.warning(
                "Adding {:.2f} mL of {} exceeds remaining capacity {:.2f} mL "
                "in container {}",
                substance.volume_ml,
                substance.name,
                self.remaining_capacity_ml,
                self.usd_path,
            )

        existing = self.get_substance(substance.name)
        if existing is not None:
            total_vol = existing.volume_ml + substance.volume_ml
            existing.concentration_mol_l = (
                existing.concentration_mol_l * existing.volume_ml
                + substance.concentration_mol_l * substance.volume_ml
            ) / total_vol
            existing.temperature_c = (
                existing.temperature_c * existing.volume_ml
                + substance.temperature_c * substance.volume_ml
            ) / total_vol
            existing.volume_ml = total_vol
        else:
            self._substances.append(substance)

        logger.info(
            "Container {} now holds {:.2f}/{:.2f} mL",
            self.usd_path,
            self.total_volume_ml,
            self.capacity_ml,
        )

    def remove_substance(
        self, name: str, volume_ml: float
    ) -> SubstanceInstance:
        """Remove a given volume of a substance and return the removed portion.

        Raises:
            ValueError: If the substance is not found or insufficient volume.
        """
        existing = self.get_substance(name)
        if existing is None:
            raise ValueError(
                f"Substance '{name}' not found in container {self.usd_path}"
            )
        if volume_ml > existing.volume_ml + 1e-6:
            raise ValueError(
                f"Cannot remove {volume_ml:.2f} mL of '{name}' — "
                f"only {existing.volume_ml:.2f} mL available"
            )

        volume_ml = min(volume_ml, existing.volume_ml)
        removed = SubstanceInstance(
            property=existing.property,
            volume_ml=volume_ml,
            concentration_mol_l=existing.concentration_mol_l,
            temperature_c=existing.temperature_c,
        )
        existing.volume_ml -= volume_ml

        if existing.volume_ml < 1e-9:
            self._substances = [s for s in self._substances if s.name != name]

        logger.info(
            "Removed {:.2f} mL of {} from {}",
            volume_ml,
            name,
            self.usd_path,
        )
        return removed

    def set_temperature(self, temperature_c: float) -> None:
        """Set the temperature of all substances in this container."""
        for s in self._substances:
            s.temperature_c = temperature_c
        logger.info(
            "Set temperature of {} to {:.1f} °C",
            self.usd_path,
            temperature_c,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize container state to dictionary."""
        return {
            "usd_path": self.usd_path,
            "container_type": self.container_type,
            "capacity_ml": self.capacity_ml,
            "substances": [s.to_dict() for s in self._substances],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Container":
        """Deserialize container from dictionary."""
        container = cls(
            usd_path=data["usd_path"],
            container_type=data["container_type"],
            capacity_ml=data["capacity_ml"],
        )
        for s_data in data.get("substances", []):
            container._substances.append(SubstanceInstance.from_dict(s_data))
        return container
