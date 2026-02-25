from .single_object_task import SingleObjectTask


class ShakeTask(SingleObjectTask):
    """Shake-beaker task: single object with a 2000-step episode limit."""

    def step(self):
        self.frame_idx += 1
        if not self.check_frame_limits(max_steps=2000):
            return None
        return self.get_basic_state_info(object_path=self.current_obj_path)
