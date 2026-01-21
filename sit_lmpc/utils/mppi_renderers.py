import jax.numpy as jnp
import numpy as np

class MPPIRenderers:
    def __init__(self, envrenderer, track, config):
        self.track = track
        self.centerline = track.centerline
        self.envrenderer = envrenderer
        self.config = config
                
    def get_point_renderer(self, points, color=(0, 0, 255), size=5, z_offset=0.01):
        """ Render points in the environment.
        """
        if points.ndim == 1:
            points = points.reshape(1, -1)
        return self.envrenderer.get_points_renderer(points, color=color, size=size, z_offset=z_offset)
        
    def set_track_renderer(self, max_points=200, boundary_color=(255, 0, 0), centerline_color=(50, 50, 50), size=2, z_offset=0.01):
        """ Render the track centerline and boundaries.
        """
        points = np.stack([self.centerline.ss, self.centerline.xs, self.centerline.ys], axis=1)
        points = points[np.arange(points.shape[0], step=points.shape[0]//max_points), :] # render only max_points points
        _ = self.envrenderer.get_closed_lines_renderer(points[:, 1:], color=centerline_color, size=size, z_offset=z_offset)
        waypoints_boundary = self.track.vmap_frenet_to_cartesian_jax(jnp.concatenate([points[:, 0:1],
                                                                jnp.ones_like(points[:, 0:1]) * self.config.half_width,
                                                                jnp.zeros_like(points[:, 0:1])], axis=1))

        _ = self.envrenderer.get_closed_lines_renderer(waypoints_boundary[:, :2], color=boundary_color, size=size, z_offset=z_offset)
        waypoints_boundary = self.track.vmap_frenet_to_cartesian_jax(jnp.concatenate([points[:, 0:1],
                                                                    jnp.ones_like(points[:, 0:1]) * -self.config.half_width,
                                                                    jnp.zeros_like(points[:, 0:1])], axis=1))
        _ = self.envrenderer.get_closed_lines_renderer(waypoints_boundary[:, :2], color=boundary_color, size=size, z_offset=z_offset)
