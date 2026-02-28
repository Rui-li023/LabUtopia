"""Substance registry with pre-defined common lab chemicals."""

from typing import Dict, Optional

from loguru import logger

from chemistry.substance import SubstanceInstance, SubstanceProperty


class SubstanceRegistry:
    """Registry of known chemical substances.

    Follows the same registry pattern used by task/controller factories.
    """

    _registry: Dict[str, SubstanceProperty] = {}

    @classmethod
    def register(cls, prop: SubstanceProperty) -> None:
        """Register a substance property definition."""
        if prop.name in cls._registry:
            logger.warning("Overwriting registered substance: {}", prop.name)
        cls._registry[prop.name] = prop
        logger.info("Registered substance: {}", prop.name)

    @classmethod
    def get(cls, name: str) -> SubstanceProperty:
        """Look up a substance by name.

        Raises:
            KeyError: If the substance is not registered.
        """
        if name not in cls._registry:
            raise KeyError(
                f"Substance '{name}' not registered. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def get_or_none(cls, name: str) -> Optional[SubstanceProperty]:
        """Look up a substance by name, returning None if not found."""
        return cls._registry.get(name)

    @classmethod
    def create_instance(
        cls,
        name: str,
        volume_ml: float,
        concentration_mol_l: float = 0.0,
        temperature_c: float = 25.0,
    ) -> SubstanceInstance:
        """Create a SubstanceInstance from a registered substance name."""
        prop = cls.get(name)
        return SubstanceInstance(
            property=prop,
            volume_ml=volume_ml,
            concentration_mol_l=concentration_mol_l,
            temperature_c=temperature_c,
        )

    @classmethod
    def list_substances(cls) -> list:
        """Return names of all registered substances."""
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Remove all registered substances (useful for testing)."""
        cls._registry.clear()


# ------------------------------------------------------------------
# Pre-register common laboratory chemicals
# ------------------------------------------------------------------

_BUILTIN_SUBSTANCES = [
    SubstanceProperty(
        name="water",
        formula="H2O",
        molecular_weight=18.015,
        state="liquid",
        density=1.0,
        boiling_point=100.0,
        melting_point=0.0,
        color=(0.7, 0.85, 1.0, 0.4),
    ),
    SubstanceProperty(
        name="hydrochloric_acid",
        formula="HCl",
        molecular_weight=36.461,
        state="liquid",
        density=1.19,
        boiling_point=-85.1,
        melting_point=-114.2,
        color=(0.9, 0.95, 1.0, 0.4),
    ),
    SubstanceProperty(
        name="sodium_hydroxide",
        formula="NaOH",
        molecular_weight=39.997,
        state="solid",
        density=2.13,
        boiling_point=1388.0,
        melting_point=323.0,
        color=(1.0, 1.0, 1.0, 1.0),
    ),
    SubstanceProperty(
        name="ethanol",
        formula="C2H5OH",
        molecular_weight=46.069,
        state="liquid",
        density=0.789,
        boiling_point=78.37,
        melting_point=-114.1,
        color=(0.95, 0.95, 0.95, 0.3),
    ),
    SubstanceProperty(
        name="sulfuric_acid",
        formula="H2SO4",
        molecular_weight=98.079,
        state="liquid",
        density=1.83,
        boiling_point=337.0,
        melting_point=10.31,
        color=(0.9, 0.9, 0.85, 0.5),
    ),
    SubstanceProperty(
        name="sodium_chloride",
        formula="NaCl",
        molecular_weight=58.44,
        state="solid",
        density=2.16,
        boiling_point=1465.0,
        melting_point=801.0,
        color=(1.0, 1.0, 1.0, 1.0),
    ),
    SubstanceProperty(
        name="phenolphthalein",
        formula="C20H14O4",
        molecular_weight=318.33,
        state="solid",
        density=1.277,
        boiling_point=557.0,
        melting_point=262.5,
        color=(1.0, 1.0, 1.0, 1.0),
    ),
    SubstanceProperty(
        name="acetic_acid",
        formula="CH3COOH",
        molecular_weight=60.052,
        state="liquid",
        density=1.049,
        boiling_point=117.9,
        melting_point=16.6,
        color=(0.95, 0.95, 0.95, 0.3),
    ),
]

for _prop in _BUILTIN_SUBSTANCES:
    SubstanceRegistry.register(_prop)
