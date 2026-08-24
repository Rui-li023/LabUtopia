import numpy as np
from typing import Optional
from scipy.spatial.transform import Rotation as R

from .base_controller import BaseController
from .atomic_actions.pick_controller import PickController
from .atomic_actions.place_controller import PlaceController
from .atomic_actions.pour_controller import PourController
from .atomic_actions.shake_controller import ShakeController
from utils.task_utils import TaskUtils

class CleanBeakerTaskController(BaseController):
    """
    Controller for clean beaker tasks with two operation modes:
    - Collection mode: Gathers training data through demonstrations
    - Inference mode: Executes learned policies for autonomous cleaning

    Attributes:
        mode (str): Operation mode ("collect" or "infer")
        _current_step (int): Current step in the task sequence
        frame_count (int): Frame counter for episode management
    """
    
    def __init__(self, cfg, robot):
        # BaseController.__init__ already dispatches to the correct
        # _init_{collect,replay,infer}_mode for cfg.mode. The old code
        # re-dispatched here with a 2-way `collect / else infer` branch, which
        # in replay mode wrongly built an inference engine (loading a missing
        # checkpoint config) and crashed. Let the base handle the dispatch.
        super().__init__(cfg, robot)
        self._current_step = 1
        self.frame_count = 0
        # Init success-gate peak trackers here too (not only in reset()) so a
        # step() before the first reset() (collect frame 0) can't AttributeError.
        self._z_init_b1 = None; self._zmax_b1 = None
        self._z_init_b2 = None; self._zmax_b2 = None
        self._quat_init_b1 = None; self._quat_init_b2 = None
        self._tilt_peak_b1 = 0.0; self._tilt_peak_b2 = 0.0
        # Oracle phase-advance state for INFER mode (device_operate pattern).
        # success_steps records which step sub-goals have been detected; the
        # shake accumulator tracks beaker1 XY path during step 5. Init here (not
        # only in reset) so a step() before the first reset() can't AttributeError.
        self.success_steps = set()
        self._shake_path = 0.0
        self._b1_prev_xy = None
    
    def _init_collect_mode(self, cfg, robot):
        """
        Initializes components for data collection mode.
        Sets up atomic action controllers and data collector.

        Args:
            cfg: Configuration object containing collection settings
            robot: Robot instance to control
        """
        super()._init_collect_mode(cfg, robot)

        # 1. Pick beaker2
        self.pick_beaker2 = PickController(
            name="pick_beaker2",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 1, 0.05, 0.004, 1]
        )

        # 2. Pour beaker2 to beaker1
        # Position-controlled pour (see pick_pour_controller for the full
        # rationale): the velocity pour recorded the integrated commanded
        # velocity as the wrist action, which sat 13-24 deg BELOW the measured
        # state across the whole return leg, so a closed-loop policy is never
        # shown a "raise the wrist" command. Position pour records the command
        # it actually sends and parks upright at the end.
        self.pour_beaker2 = PourController(
            name="pour_beaker2",
            cspace_controller=self.rmp_controller,
            # events_dt[5] was 1 (a single frame): with a position pour that is the
            # hold-upright segment, so give it ~100 frames of a static upright pose.
            events_dt=[0.006, 0.005, 0.009, 0.05, 0.009, 0.01],
            position_pour=bool(getattr(getattr(cfg, "task", None), "position_pour", True)),
            pour_angle_rad=float(getattr(getattr(cfg, "task", None), "pour_angle_rad", 1.2)),
        )

        # 3. Place beaker2 to plat2
        self.place_beaker2 = PlaceController(
            name="place_beaker2",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.003, 0.008, 1, 0.05, 0.01, 1],
            robot=robot,
        )

        # 4. Pick beaker1
        self.pick_beaker1 = PickController(
            name="pick_beaker1",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 1, 0.05, 0.004, 1]
        )

        # 5. Shake beaker1
        self.shake_beaker1 = ShakeController(
            name="shake_beaker1",
            cspace_controller=self.rmp_controller
        )

        # 6. Pour beaker1 to target_beaker
        # Position-controlled pour (see pick_pour_controller for the full
        # rationale): the velocity pour recorded the integrated commanded
        # velocity as the wrist action, which sat 13-24 deg BELOW the measured
        # state across the whole return leg, so a closed-loop policy is never
        # shown a "raise the wrist" command. Position pour records the command
        # it actually sends and parks upright at the end.
        self.pour_beaker1 = PourController(
            name="pour_beaker1",
            cspace_controller=self.rmp_controller,
            # events_dt[5] was 1 (a single frame): with a position pour that is the
            # hold-upright segment, so give it ~100 frames of a static upright pose.
            events_dt=[0.006, 0.005, 0.009, 0.05, 0.009, 0.01],
            position_pour=bool(getattr(getattr(cfg, "task", None), "position_pour", True)),
            pour_angle_rad=float(getattr(getattr(cfg, "task", None), "pour_angle_rad", 1.2)),
        )

        # 7. Place beaker1 to plat1
        self.place_beaker1 = PlaceController(
            name="place_beaker1",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.003, 0.008, 1, 0.05, 0.01, 1],
            robot=robot,
        )

    def reset(self):
        super().reset()
        
        if self.mode == "collect":
            self.pick_beaker2.reset()
            self.pour_beaker2.reset()
            self.place_beaker2.reset()
            self.pick_beaker1.reset()
            self.shake_beaker1.reset()
            self.pour_beaker1.reset()
            self.place_beaker1.reset()
        elif self.mode == "infer":
            self.inference_engine.reset()
        
        self._current_step = 1
        self.frame_count = 0
        self._logged_step_1 = False
        # Mode-safe peak trackers for the success gates (lift + pour tilt),
        # reset per episode so stale values can't leak. Updated every frame in
        # step() (ALL modes — phases never advance in replay, so success gates
        # must never depend on the collect-only atomic controllers).
        self._z_init_b1 = None; self._zmax_b1 = None
        self._z_init_b2 = None; self._zmax_b2 = None
        self._quat_init_b1 = None; self._quat_init_b2 = None
        self._tilt_peak_b1 = 0.0; self._tilt_peak_b2 = 0.0
        # Reset oracle phase-advance state every episode (infer). _current_step
        # is already reset to 1 above; success_steps / shake accumulator too.
        self.success_steps = set()
        self._shake_path = 0.0
        self._b1_prev_xy = None

    def _track_success_signals(self, state) -> None:
        """Accumulate peak lift + peak tilt for both beakers every frame (all
        modes). _check_success reads these; they use only all-mode signals
        (object pose/quat), never atomic-controller internals."""
        if not isinstance(state, dict):
            return
        tu = TaskUtils.get_instance()
        p1 = state.get('beaker_1_position')
        if p1 is not None:
            z = float(p1[2])
            if self._z_init_b1 is None:
                self._z_init_b1 = z; self._zmax_b1 = z
            self._zmax_b1 = max(self._zmax_b1, z)
        # Live orientation: the rigid body (PhysicsRigidBodyAPI) is on the /mesh
        # subprim — physics updates ITS xformOp, not the top-level prim's (which
        # stays at the authored value). Mirrors pick_pour_task's source_quaternion.
        q1 = self.object_utils.get_transform_quat(object_path=self.cfg.beaker_1 + "/mesh")
        if q1 is not None:
            if self._quat_init_b1 is None:
                self._quat_init_b1 = q1
            self._tilt_peak_b1 = max(self._tilt_peak_b1, tu.rotation_angle_deg(self._quat_init_b1, q1))
        p2 = state.get('beaker_2_position')
        if p2 is not None:
            z = float(p2[2])
            if self._z_init_b2 is None:
                self._z_init_b2 = z; self._zmax_b2 = z
            self._zmax_b2 = max(self._zmax_b2, z)
        q2 = self.object_utils.get_transform_quat(object_path=self.cfg.beaker_2 + "/mesh")
        if q2 is not None:
            if self._quat_init_b2 is None:
                self._quat_init_b2 = q2
            self._tilt_peak_b2 = max(self._tilt_peak_b2, tu.rotation_angle_deg(self._quat_init_b2, q2))
        # Oracle step-5 shake detector: accumulate beaker1's XY path length only
        # while step 5 is the active sub-goal. _track_success_signals is called
        # from step() BEFORE the mode dispatch, so this updates every frame in all
        # modes. A real 3-cycle lateral shake (amplitude 0.06-0.14 m) travels
        # ~0.3-0.6 m; a static hold ~0. Resetting _b1_prev_xy to None outside
        # step 5 keeps transport/place motion from inflating the accumulator.
        # p1 was bound above as state.get('beaker_1_position').
        if p1 is not None:
            xy = np.array([float(p1[0]), float(p1[1])])
            if self._current_step == 5:
                if self._b1_prev_xy is not None:
                    self._shake_path += float(np.linalg.norm(xy - self._b1_prev_xy))
                self._b1_prev_xy = xy
            else:
                self._b1_prev_xy = None

    def step(self, state):
        # _step_replay (base) reads self.state in its diagnostics and
        # _check_success; this override bypasses BaseController.step, so set it
        # here too (mirrors the base contract).
        self.state = state
        self._track_success_signals(state)
        if self.mode == "collect":
            return self._step_collect(state)
        elif self.mode == "replay":
            return self._step_replay(state)
        else:
            return self._step_infer(state)
    
    def _step_collect(self, state):
        """
        Executes one step in collection mode.
        Records demonstrations and manages episode transitions.

        Args:
            state (dict): Current environment state

        Returns:
            tuple: (action, done, success) indicating control output and episode status
        """
        action = None
        record_array = None
        done = False
        success = False

        if self._current_step == 1:
            if not getattr(self, "_logged_step_1", False):
                print(f"[cleanbeaker] step 1 (pick beaker2) begin; beaker_2_pos={state['beaker_2_position'].tolist() if hasattr(state['beaker_2_position'], 'tolist') else state['beaker_2_position']}")
                self._logged_step_1 = True
            # 1. Pick beaker2
            action, record_array = self.pick_beaker2.forward(
                picking_position=state['beaker_2_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name="beaker_l",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                pre_offset_x=0.05,
                pre_offset_z=0.05,
                # 0.024: the least-bad point in a razor-thin window. 0.027 = zero
                # squeeze → replay slip (0%); 0.022/0.020 = 2-4 mm squeeze →
                # EJECTS in collect (rate 5%/0%). 0.024 gives collect ~71% and
                # replay ~80% (beaker2 still slips ~2/10 — the open-loop pour
                # can't be held tighter without ejecting). See delivery report.
                gripper_distances=0.024
            )
            if self.pick_beaker2.is_done():
                print("[cleanbeaker] step 1 (pick beaker2) done")
                self._current_step = 2

        elif self._current_step == 2:
            # 2. Pour beaker2 to beaker1
            action, record_array = self.pour_beaker2.forward(
                articulation_controller=self.robot.get_articulation_controller(),
                current_joint_positions=self.robot.get_joint_positions(),
                source_size=state['object_size'],
                target_position=state['beaker_1_position'],
                gripper_position=state['gripper_position'],
                source_name="beaker",
                current_joint_velocities=self.robot.get_joint_velocities(),
                pour_speed=-1,
            )
            if self.pour_beaker2.is_done():
                print(f"[cleanbeaker] step 2 (pour beaker2→beaker1) done; beaker2_pos={state['beaker_2_position'].tolist() if hasattr(state['beaker_2_position'], 'tolist') else state['beaker_2_position']}")
                self._current_step = 3

        elif self._current_step == 3:
            # 3. Place beaker2 to plat2
            # Aim the plat centre with the stock 0.027 grip: the released
            # beaker always settles FLAT at dx=0.0403±0.0003 (dz=0.0358, fully
            # on the platform). Every attempt to cancel that settle shift —
            # offset aim points (-0.01/-0.02/-0.04) or a firmer grip (0.024) —
            # moved the drop dynamics off the flat sweet spot and the beaker
            # tipped on the plat edge 40-50% of the time (dz 0.049-0.056).
            # The deterministic flat landing is the right behaviour; the
            # success box in _check_success accounts for the settle shift.
            action, record_array = self.place_beaker2.forward(
                place_position=state['plat_2_position'],
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 40])).as_quat(),
                gripper_position=state['gripper_position']
            )
            if self.place_beaker2.is_done():
                print(f"[cleanbeaker] step 3 (place beaker2→plat2) done; beaker2_pos={state['beaker_2_position'].tolist() if hasattr(state['beaker_2_position'], 'tolist') else state['beaker_2_position']} plat2={state['plat_2_position'].tolist() if hasattr(state['plat_2_position'], 'tolist') else state['plat_2_position']}")
                self._current_step = 4

        elif self._current_step == 4:
            # 4. Pick beaker1
            action, record_array = self.pick_beaker1.forward(
                picking_position=state['beaker_1_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name="beaker_l",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                # 0.024 (was 0.027): same squeeze margin as beaker2 — beaker1
                # slipped during the replayed shake/pour (ended 22 cm off plat1).
                gripper_distances=0.024
            )
            if self.pick_beaker1.is_done():
                print(f"[cleanbeaker] step 4 (pick beaker1) done; beaker1_pos={state['beaker_1_position'].tolist() if hasattr(state['beaker_1_position'], 'tolist') else state['beaker_1_position']}")
                self._current_step = 5

        elif self._current_step == 5:
            # 5. Shake beaker1
            action, record_array = self.shake_beaker1.forward(
                current_joint_positions=self.robot.get_joint_positions(),
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
            )
            if self.shake_beaker1.is_done():
                print("[cleanbeaker] step 5 (shake beaker1) done")
                self._current_step = 6

        elif self._current_step == 6:
            # 6. Pour beaker1 to target_beaker
            action, record_array = self.pour_beaker1.forward(
                articulation_controller=self.robot.get_articulation_controller(),
                current_joint_positions=self.robot.get_joint_positions(),
                source_size=state['object_size'],
                source_name="beaker",
                target_position=state['target_position'],
                gripper_position=state['gripper_position'],
                current_joint_velocities=self.robot.get_joint_velocities(),
                pour_speed=-1,
            )
            if self.pour_beaker1.is_done():
                print(f"[cleanbeaker] step 6 (pour beaker1→target) done; beaker1_pos={state['beaker_1_position'].tolist() if hasattr(state['beaker_1_position'], 'tolist') else state['beaker_1_position']}")
                self._current_step = 7

        elif self._current_step == 7:
            # 7. Place beaker1 to plat1. Uncompensated: with the shallower
            # 10° place yaw its settle shift stays under the 0.04 tolerance
            # (passed consistently pre-fix); see step 3 for the beaker2 story.
            action, record_array = self.place_beaker1.forward(
                place_position=state['plat_1_position'],
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                gripper_position=state['gripper_position']
            )
            if self.place_beaker1.is_done():
                print(f"[cleanbeaker] step 7 (place beaker1→plat1) done; beaker1_pos={state['beaker_1_position'].tolist() if hasattr(state['beaker_1_position'], 'tolist') else state['beaker_1_position']} plat1={state['plat_1_position'].tolist() if hasattr(state['plat_1_position'], 'tolist') else state['plat_1_position']}")
                success = self._check_success()
                if success:
                    self._last_failure_reason = ""
                    self.data_collector.write_cached_data(state['joint_positions'][:-1])
                    self._last_success = True
                else:
                    self._last_failure_reason = "Beaker placement check failed: beakers not on target platforms"
                    self.data_collector.clear_cache()
                    self._last_success = False
                done = True
                self.reset_needed = True
                action = None

        if not done and 'camera_data' in state and record_array is not None:
            self.data_collector.cache_step(
                camera_images=state['camera_data'],
                joint_angles=state['joint_positions'][:-1],
                action=record_array,
                language_instruction=self.get_language_instruction()
            )

        return action, done, success
    
    def _check_step_success(self, state) -> bool:
        """Oracle sub-goal predicate for the CURRENT step (infer mode).

        Reads ONLY live world state: object geometry centres (state[...]), the
        beaker's live /mesh quaternion, and the all-mode peak trackers updated
        every frame in _track_success_signals() BEFORE the mode dispatch. No
        object binding / teleport / atomic-controller internals, so it is valid
        while the POLICY drives the robot. Returns True iff the current step's
        sub-goal is reached.
        """
        if not isinstance(state, dict):
            return False
        tu = TaskUtils.get_instance()
        b1 = state.get('beaker_1_position')
        b2 = state.get('beaker_2_position')
        p1 = state.get('plat_1_position')
        p2 = state.get('plat_2_position')

        def cur_tilt(beaker_path, quat_init):
            if quat_init is None:
                return 0.0
            q = self.object_utils.get_transform_quat(object_path=beaker_path + "/mesh")
            if q is None:
                return 0.0
            return tu.rotation_angle_deg(quat_init, q)

        def on_plat(b, p):
            # Relaxed vs the strict terminal _check_success gate so a POLICY
            # place still registers, but still far inside the >0.14 m initial
            # beaker<->plat separation (no premature trip) and requiring the
            # beaker to be SET DOWN (0 < dz < 0.10), not hovering above it.
            if b is None or p is None:
                return False
            dxy = float(np.linalg.norm(np.asarray(b[:2], dtype=float)
                                       - np.asarray(p[:2], dtype=float)))
            dz = float(b[2]) - float(p[2])
            return dxy < 0.08 and 0.0 < dz < 0.10

        step = self._current_step
        if step == 1:                    # pick beaker2: lifted clear of table
            if b2 is None or self._z_init_b2 is None:
                return False
            return float(b2[2]) - self._z_init_b2 > 0.05
        if step == 2:                    # pour beaker2 -> beaker1 (tilt then upright)
            return (self._tilt_peak_b2 > 30.0
                    and cur_tilt(self.cfg.beaker_2, self._quat_init_b2) < 25.0)
        if step == 3:                    # place beaker2 on plat2
            return on_plat(b2, p2)
        if step == 4:                    # pick beaker1: lifted clear of table
            if b1 is None or self._z_init_b1 is None:
                return False
            return float(b1[2]) - self._z_init_b1 > 0.05
        if step == 5:                    # shake beaker1 (lateral oscillation)
            if b1 is None or self._z_init_b1 is None:
                return False
            aloft = float(b1[2]) - self._z_init_b1 > 0.05
            return aloft and self._shake_path > 0.15
        if step == 6:                    # pour beaker1 -> target (tilt then upright)
            return (self._tilt_peak_b1 > 30.0
                    and cur_tilt(self.cfg.beaker_1, self._quat_init_b1) < 25.0)
        if step == 7:                    # place beaker1 on plat1 (terminal)
            return on_plat(b1, p1)
        return False

    def _step_infer(self, state):
        """Infer mode with ORACLE phase advancement (device_operate pattern).

        The POLICY drives the robot; this controller only OBSERVES live world
        state to decide when the current sub-goal (step) is reached, records it
        in success_steps, then feeds the NEXT step's language instruction to the
        policy. Success: every step's predicate passed IN ORDER, i.e. all 7
        steps land in success_steps and self._current_step reaches the FINISHED
        sentinel (8). Trackers the predicates read (_z_init_*, _quat_init_*,
        _tilt_peak_*, _shake_path) are updated in step()->_track_success_signals
        BEFORE this dispatch, so they are live in infer.
        """
        # 1) Detect completion of the CURRENT sub-goal, record it, advance.
        if self._current_step <= 7 and self._check_step_success(state):
            self.success_steps.add(self._current_step)
            print(f"[cleanbeaker infer] step {self._current_step} success! "
                  f"-> advance (done={sorted(self.success_steps)})")
            self._current_step += 1

        # 2) Terminal: all seven sub-goals reached in order.
        if self._current_step > 7:
            self.reset_needed = True
            self._last_success = len(self.success_steps) == 7
            if self._last_success:
                self._last_failure_reason = ""
            else:
                self._last_failure_reason = (
                    f"clean_beaker infer ended with steps "
                    f"{sorted(self.success_steps)} (need 1..7)")
            return None, True, self._last_success

        # 3) Otherwise keep driving the policy with the current step's instruction.
        language_instruction = self.get_language_instruction()
        if language_instruction is not None:
            state['language_instruction'] = language_instruction
        else:
            state['language_instruction'] = "Clean the beakers"

        action = self.inference_engine.step_inference(state)
        return action, False, False

    def _check_success(self):
        # Use world-space geometry centres so the check is independent of
        # how each prim's xform is parented or stacked. The previous code
        # mixed mesh-local "/mesh" xform with parent plat xform, which gave
        # inconsistent z values across episodes.
        beaker1_pos = self.object_utils.get_geometry_center(object_path=self.cfg.beaker_1)
        beaker2_pos = self.object_utils.get_geometry_center(object_path=self.cfg.beaker_2)
        plat1_pos   = self.object_utils.get_geometry_center(object_path=self.cfg.plat_1)
        plat2_pos   = self.object_utils.get_geometry_center(object_path=self.cfg.plat_2)

        if beaker1_pos is None or beaker2_pos is None or plat1_pos is None or plat2_pos is None:
            return False

        def beaker_on_plat(b, p, label):
            dx, dy, dz = abs(b[0] - p[0]), abs(b[1] - p[1]), b[2] - p[2]
            # dx tolerance 0.05 (was 0.04): the side-grasped beaker
            # deterministically settles flat at dx=0.0403 from the EE target
            # (grip-tilt roll on release); the platform is >0.13 m wide, so
            # the beaker is fully on it. 0.04 sat exactly ON the settle
            # point and flipped episodes on sub-millimetre physics noise.
            ok = dx < 0.05 and dy < 0.04 and 0.0 < dz < 0.08
            # Print on success too: the margins tell us how close each place
            # runs to the tolerance (used to calibrate the settle offsets).
            print(
                f"[cleanbeaker debug] {label} {'ok' if ok else 'fail'}: "
                f"dx={dx:.4f} dy={dy:.4f} dz={dz:.4f}"
            )
            return ok

        ok1 = beaker_on_plat(beaker1_pos, plat1_pos, "beaker1↔plat1")
        ok2 = beaker_on_plat(beaker2_pos, plat2_pos, "beaker2↔plat2")
        if not (ok1 and ok2):
            return False

        # Pour-task certification (mode-safe peak trackers from step()): each
        # beaker must have been (a) LIFTED clear of the table — rejects a beaker
        # that drifted onto a plat without being grasped — AND (b) actually
        # POURED, i.e. tilted >30°. Calibrated from replay data: a real scripted
        # pour reaches ~50° on both beakers, while transport keeps them upright
        # (<30°), so 30° cleanly separates pour from carry. Both gates read only
        # all-mode signals (object pose/quat + step() peak trackers), never the
        # collect-only atomic controllers.
        if self._z_init_b1 is None or self._z_init_b2 is None:
            return False
        lift1 = self._zmax_b1 - self._z_init_b1
        lift2 = self._zmax_b2 - self._z_init_b2
        ok = (lift1 > 0.05 and lift2 > 0.05
              and self._tilt_peak_b1 > 30.0 and self._tilt_peak_b2 > 30.0)
        print(f"[cleanbeaker debug] lift1={lift1:.3f} lift2={lift2:.3f} "
              f"tilt1={self._tilt_peak_b1:.0f} tilt2={self._tilt_peak_b2:.0f} "
              f"-> {'OK' if ok else 'FAIL'}")
        return ok
    
    def is_success(self):
        """Task success: every step's sub-goal detected in order (device_operate style)."""
        return len(self.success_steps) == 7

    def get_language_instruction(self) -> Optional[str]:
        step_instructions = {
            1: ('Pick up the second beaker', 'Pick up the second beaker from the table and lift it clear of the surface'),
            2: ('Pour the second beaker into the first beaker', 'Pour the contents of the second beaker into the first beaker carefully'),
            3: ('Place the second beaker on the second platform', 'Move the second beaker to the second platform and set it down carefully'),
            4: ('Pick up the first beaker', 'Pick up the first beaker from the table and lift it clear of the surface'),
            5: ('Shake the first beaker', 'Shake the first beaker to mix the contents thoroughly'),
            6: ('Pour the first beaker into the target beaker', 'Pour the contents of the first beaker into the target beaker carefully'),
            7: ('Place the first beaker on the first platform', 'Move the first beaker to the first platform and set it down carefully'),
        }
        direct, detailed = step_instructions.get(
            self._current_step,
            ('Clean the beakers', 'Complete the current cleaning step carefully'),
        )
        return self._get_cached_instruction(
            f"cleanbeaker:{self._current_step}",
            self._build_instruction_templates(direct, detailed),
        )
