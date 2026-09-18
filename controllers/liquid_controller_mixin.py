"""All-mode liquid gates for composite controllers; no scene initialization."""

from typing import Any

from isaacsim.core.utils.types import ArticulationAction

from controllers.atomic_actions.liquid_pour_controller import LiquidPourController
from utils.liquid_metrics import liquid_enabled
from utils.liquid_transfers import LiquidRouteMonitor


class _RoutedPourController(LiquidPourController):
    def __init__(self, name: str, owner: Any, route: str) -> None:
        self.owner, self.route = owner, route
        super().__init__(name, owner.rmp_controller, owner.cfg)

    def forward(self, **kwargs: Any) -> Any:
        # Legacy call sites supply geometric guesses; liquid mode instead uses
        # the measured source/receiver mouths, without moving either object.
        state = dict(self.owner.state)
        state["liquid"] = state["liquid_transfers"][self.route]
        return super().forward(state)


class LiquidControllerMixin:
    LIQUID_POUR_CONTROLLERS: dict[str, str] = {}

    def __init__(self, cfg: Any, robot: Any) -> None:
        self.liquid_monitor = LiquidRouteMonitor(cfg) if liquid_enabled(cfg) else None
        super().__init__(cfg, robot)

    def _init_collect_mode(self, cfg: Any, robot: Any) -> None:
        super()._init_collect_mode(cfg, robot)
        if liquid_enabled(cfg):
            for name, route in self.LIQUID_POUR_CONTROLLERS.items():
                setattr(self, name, _RoutedPourController(name, self, route))
            if self.LIQUID_POUR_CONTROLLERS and hasattr(self, "_switch_active_controller"):
                self._switch_active_controller()

    def reset(self) -> None:
        if self.liquid_monitor is not None:
            self.liquid_monitor.reset()
        super().reset()

    def step(self, state: dict[str, Any]) -> Any:
        if self.liquid_monitor is not None:
            self.liquid_monitor.update(state)
        return super().step(state)

    def liquid_success(self) -> bool:
        return self.liquid_monitor is None or self.liquid_monitor.success()

    def record_liquid_handoff(self, state: dict[str, Any]) -> None:
        if self.liquid_monitor is not None and "camera_data" in state:
            hold = ArticulationAction(joint_positions=[None] * len(state["joint_positions"]))
            self.data_collector.cache_step(
                camera_images=state["camera_data"],
                joint_angles=state["joint_positions"][:-1],
                action=self.active_controller._build_record_array(hold, state["joint_positions"]),
                language_instruction=self.get_language_instruction(),
            )

    def _check_success(self) -> bool:
        return self.liquid_success() and super()._check_success()
