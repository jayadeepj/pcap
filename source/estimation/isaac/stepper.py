class Stepper:
    def __init__(self, gym, sim, viewer):
        self.frame_count = 0
        self.gym = gym
        self.sim = sim
        self.viewer = viewer

    def step_through(self, steps=1):

        def _finish(msg):
            print(msg)
            if self.viewer:
                self.gym.destroy_viewer(self.viewer)

            self.gym.destroy_sim(self.sim)
            raise StopExecution

        if steps == -1:
            _finish("User Request: Exiting simulation")

        for _ in range(steps):
            if self.viewer and self.gym.query_viewer_has_closed(self.viewer):
                self.gym.destroy_viewer(self.viewer)
                _finish("Viewer Closed: Exiting simulation")

            if self.frame_count == 1e10:
                _finish("Max Steps reached: Exiting simulation")

            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.frame_count += 1

            # update viewer if enabled
            if self.viewer:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, False)
                self.gym.sync_frame_time(self.sim)


class StopExecution(Exception):
    def _render_traceback_(self):
        pass
