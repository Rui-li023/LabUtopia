"""Global chemistry state manager for a simulation scene."""

from typing import Any, Dict, List, Optional

from loguru import logger

from chemistry.container import Container
from chemistry.substance import SubstanceInstance


class LabChemistryState:
    """Manages chemical state for all containers in a scene.

    Provides container lookup by USD prim path, bulk transfer operations,
    and full serialization for episode recording.
    """

    def __init__(self) -> None:
        self._containers: Dict[str, Container] = {}  # usd_path -> Container

    # ------------------------------------------------------------------
    # Container management
    # ------------------------------------------------------------------

    def add_container(self, container: Container) -> None:
        """Register a container in the scene."""
        if container.usd_path in self._containers:
            logger.warning(
                "Overwriting container at {}", container.usd_path
            )
        self._containers[container.usd_path] = container
        logger.info(
            "Added container {} (type={}, capacity={:.0f} mL)",
            container.usd_path,
            container.container_type,
            container.capacity_ml,
        )

    def get_container(self, usd_path: str) -> Optional[Container]:
        """Look up a container by its USD prim path."""
        return self._containers.get(usd_path)

    def remove_container(self, usd_path: str) -> None:
        """Remove a container from the scene."""
        if usd_path in self._containers:
            del self._containers[usd_path]
            logger.info("Removed container {}", usd_path)
        else:
            logger.warning(
                "Cannot remove container {} — not found", usd_path
            )

    @property
    def containers(self) -> List[Container]:
        """All registered containers."""
        return list(self._containers.values())

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def transfer(
        self,
        src_path: str,
        dst_path: str,
        substance_name: str,
        volume_ml: float,
    ) -> None:
        """Transfer a substance from one container to another (e.g. pouring).

        Raises:
            KeyError: If either container is not registered.
            ValueError: If the substance is not found or insufficient volume.
        """
        src = self._containers.get(src_path)
        if src is None:
            raise KeyError(f"Source container '{src_path}' not found")

        dst = self._containers.get(dst_path)
        if dst is None:
            raise KeyError(f"Destination container '{dst_path}' not found")

        removed = src.remove_substance(substance_name, volume_ml)
        dst.add_substance(removed)
        logger.info(
            "Transferred {:.2f} mL of {} from {} to {}",
            volume_ml,
            substance_name,
            src_path,
            dst_path,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full chemistry state to a dictionary."""
        return {
            "containers": {
                path: c.to_dict() for path, c in self._containers.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LabChemistryState":
        """Deserialize from a dictionary."""
        state = cls()
        for _path, c_data in data.get("containers", {}).items():
            container = Container.from_dict(c_data)
            state._containers[container.usd_path] = container
        return state

    def reset(self) -> None:
        """Clear all chemistry state."""
        self._containers.clear()
        logger.info("Chemistry state reset")
