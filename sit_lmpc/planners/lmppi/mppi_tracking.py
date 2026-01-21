"""An MPPI based planner."""
import jax
import jax.numpy as jnp
import os, sys

from functools import partial
import numpy as np
from numba import njit
from utilsuite import jnumpify, jax_jit

class MPPI():
    """An MPPI based planner."""
    def __init__(self, config, env, jrng, waypoints,
                 temperature=0.01, damping=0.001, track=None):
        self.config = config
        self.n_iterations = config.n_iterations
        self.n_steps = config.n_steps
        self.n_samples = config.n_samples
        self.temperature = temperature
        self.damping = damping
        # self.a_std = jnp.array(config.control_sample_std)
        self.a_std = jnp.array([1.0, 1.0])
        self.a_cov_shift = config.a_cov_shift
        self.adaptive_covariance = (config.adaptive_covariance and self.n_iterations > 1) or self.a_cov_shift
        self.a_shape = config.control_dim
        self.env = env
        self.jrng = jrng
        self.init_state(self.env, self.a_shape)
        self.accum_matrix = jnp.triu(jnp.ones((self.n_steps, self.n_steps)))
        self.track = track
        self.waypoints = waypoints
        self.diff = self.waypoints[1:, 1:3] - self.waypoints[:-1, 1:3]
        self.waypoints_distances = np.linalg.norm(self.waypoints[1:, (1, 2)] - self.waypoints[:-1, (1, 2)], axis=1)


    def init_state(self, env, a_shape):
        # uses random as a hack to support vmap
        # we should find a non-hack approach to initializing the state
        dim_a = jnp.prod(a_shape)  # np.int32
        self.env = env
        self.a_opt = 0.0*jax.random.uniform(self.jrng.new_key(), shape=(self.n_steps,
                                                dim_a))  # [n_steps, dim_a]
        # a_cov: [n_steps, dim_a, dim_a]
        if self.a_cov_shift:
            # note: should probably store factorized cov,
            # e.g. cholesky, for faster sampling
            self.a_cov = (self.a_std**2)*jnp.tile(jnp.eye(dim_a), (self.n_steps, 1, 1))
            self.a_cov_init = self.a_cov.copy()
        else:
            self.a_cov = None
            self.a_cov_init = self.a_cov
            
    
    def update(self, params, env_state, reference_traj, dyna_norm_params=None):
        self.a_opt, self.a_cov = self.shift_prev_opt(self.a_opt, self.a_cov)
        for _ in range(self.n_iterations):
            self.a_opt, self.a_cov, self.states, self.traj_opt = self.iteration_step(params, 
                                                                                     self.a_opt, 
                                                                                     self.a_cov, 
                                                                                     self.jrng.new_key(), 
                                                                                     env_state, 
                                                                                     reference_traj,
                                                                                     dyna_norm_params=dyna_norm_params)

        if self.track is not None and self.config.state_predictor in self.config.cartesian_models:
            self.states = self.convert_cartesian_to_frenet_jax(self.states)
            self.traj_opt = self.convert_cartesian_to_frenet_jax(self.traj_opt)
        self.sampled_states = self.states

    
    @partial(jax.jit, static_argnums=(0))
    def shift_prev_opt(self, a_opt, a_cov):
        a_opt = jnp.concatenate([a_opt[1:, :],
                                jnp.expand_dims(jnp.zeros((self.a_shape,)),
                                                axis=0)])  # [n_steps, a_shape]
        if self.a_cov_shift:
            a_cov = jnp.concatenate([a_cov[1:, :],
                                    jnp.expand_dims((self.a_std**2)*jnp.eye(self.a_shape),
                                                    axis=0)])
        else:
            a_cov = self.a_cov_init
        return a_opt, a_cov
    
    
    # @partial(jax.jit, static_argnums=(0))
    @jax_jit(static_argnums=(0,))
    def iteration_step(self, params, a_opt, a_cov, rng_da, env_state, reference_traj, dyna_norm_params=None):
        rng_da, rng_da_split1, rng_da_split2 = jax.random.split(rng_da, 3)
        da = jax.random.truncated_normal(
            rng_da,
            -jnp.ones_like(a_opt) * self.a_std - a_opt,
            jnp.ones_like(a_opt) * self.a_std - a_opt,
            shape=(self.n_samples, self.n_steps, self.a_shape)
        )  # [n_samples, n_steps, dim_a]

        actions = jnp.clip(jnp.expand_dims(a_opt, axis=0) + da, -1.0, 1.0)
        states = jax.vmap(self.rollout, in_axes=(0, None, None, None, None))(
            actions, env_state, params, rng_da_split1, dyna_norm_params
        )
        
        if self.config.state_predictor in self.config.cartesian_models or self.config.state_predictor == 'point_mass':
            reward = jax.vmap(self.env.reward_fn_xy, in_axes=(0, None))(
                states, reference_traj
            )
        else:
            reward = jax.vmap(self.env.reward_fn_sey, in_axes=(0, None))(
                states, reference_traj
            ) # [n_samples, n_steps]          
        
        R = jax.vmap(self.returns)(reward) # [n_samples, n_steps], pylint: disable=invalid-name
        w = jax.vmap(self.weights, 1, 1)(R)  # [n_samples, n_steps]
        da_opt = jax.vmap(jnp.average, (1, None, 1))(da, 0, w)  # [n_steps, dim_a]
        a_opt = jnp.clip(a_opt + da_opt, -1.0, 1.0)  # [n_steps, dim_a]
        if self.adaptive_covariance:
            a_cov = jax.vmap(jax.vmap(jnp.outer))(
                da, da
            )  # [n_samples, n_steps, a_shape, a_shape]
            a_cov = jax.vmap(jnp.average, (1, None, 1))(
                a_cov, 0, w
            )  # a_cov: [n_steps, a_shape, a_shape]
            a_cov = a_cov + jnp.eye(self.a_shape)*0.00001 # prevent loss of rank when one sample is heavily weighted
            
        if self.config.render:
            traj_opt = self.rollout(a_opt, env_state, params, rng_da_split2, dyna_norm_params=dyna_norm_params)
        else:
            traj_opt = states[0]
            
        return a_opt, a_cov, states, traj_opt

   
    @partial(jax.jit, static_argnums=(0))
    def returns(self, r):
        # r: [n_steps]
        return jnp.dot(self.accum_matrix, r)  # R: [n_steps]


    @partial(jax.jit, static_argnums=(0))
    def weights(self, R):  # pylint: disable=invalid-name
        # R: [n_samples]
        # R_stdzd = (R - jnp.min(R)) / ((jnp.max(R) - jnp.min(R)) + self.damping)
        # R_stdzd = R - jnp.max(R) # [n_samples] np.float32
        R_stdzd = (R - jnp.max(R)) / ((jnp.max(R) - jnp.min(R)) + self.damping)  # pylint: disable=invalid-name
        w = jnp.exp(R_stdzd / self.temperature)  # [n_samples] np.float32
        w = w/jnp.sum(w)  # [n_samples] np.float32
        return w
    
    
    # @partial(jax.jit, static_argnums=0)
    @jax_jit(static_argnums=(0,))
    def rollout(self, actions, env_state, params, rng_key, dyna_norm_params=None):
        """
        # actions: [n_steps, a_shape]
        # env: {.step(states, actions), .reward(states)}
        # env_state: np.float32
        # actions: # a_0, ..., a_{n_steps}. [n_steps, a_shape]
        # states: # s_1, ..., s_{n_steps+1}. [n_steps, env_state_shape]
        """

        def rollout_step(env_state, actions, params, rng_key, dyna_norm_params=None):
            actions = jnp.reshape(actions, self.env.a_shape)
            (env_state, env_var, mb_dyna) = self.env.step(env_state, actions, params, rng_key, dyna_norm_params=dyna_norm_params)
            return env_state
        
        states = []
        for t in range(self.n_steps):
            env_state = rollout_step(env_state, actions[t, :], params, rng_key, dyna_norm_params=dyna_norm_params)
            states.append(env_state)
            
        return jnp.asarray(states)
    
    
    # @partial(jax.jit, static_argnums=(0))
    def convert_cartesian_to_frenet_jax(self, states):
        states_shape = (*states.shape[:-1], 7)
        states = states.reshape(-1, states.shape[-1])
        converted_states = self.track.vmap_cartesian_to_frenet_jax(states[:, (0, 1, 4)])
        states_frenet = jnp.concatenate([converted_states[:, :2], 
                                         states[:, 2:4] * jnp.cos(states[:, 6:7]),
                                         converted_states[:, 2:3],
                                         states[:, 2:4] * jnp.sin(states[:, 6:7])], axis=-1)
        return states_frenet.reshape(states_shape)
    
    @partial(jax.jit, static_argnums=(0,3))    
    def get_refernece_traj_jax(self, state, target_speed, n_steps=10, DT=0.1):
        _, dist, _, _, ind = nearest_point_jax(jnp.array([state[0], state[1]]), 
                                           self.waypoints[:, (1, 2)], jnp.array(self.diff))
        
        speed = target_speed
        speeds = jnp.ones(n_steps) * speed
        
        reference = get_reference_trajectory_jax(speeds, dist, ind, 
                                            self.waypoints.copy(), int(n_steps),
                                            self.waypoints_distances.copy(), DT=DT)
        orientation = state[4]
        reference = reference.at[:, 3].set(
            jnp.where(reference[:, 3] - orientation > 5, 
                  reference[:, 3] - 2 * jnp.pi, 
                  reference[:, 3])
        )
        reference = reference.at[:, 3].set(
            jnp.where(reference[:, 3] - orientation < -5, 
                  reference[:, 3] + 2 * jnp.pi, 
                  reference[:, 3])
        )
        
        return reference
    
    def get_refernece_traj(self, state, target_speed=None, n_steps=10, DT=0.1):
        _, dist, _, _, ind = nearest_point(np.array([state[0], state[1]]), 
                                            self.waypoints[:, (1, 2)].copy())
        
        if target_speed is None:
            speed = state[3]
        else:
            speed = target_speed
        
        speeds = np.ones(n_steps) * speed
        reference = get_reference_trajectory(speeds, dist, ind, 
                                            self.waypoints, int(n_steps),
                                            self.waypoints_distances.copy(), DT=DT)
        orientation = state[4]
        reference[3, :][reference[3, :] - orientation > 5] = np.abs(
            reference[3, :][reference[3, :] - orientation > 5] - (2 * np.pi))
        reference[3, :][reference[3, :] - orientation < -5] = np.abs(
            reference[3, :][reference[3, :] - orientation < -5] + (2 * np.pi))
        
        return reference.T

@jax.jit
def get_reference_trajectory_jax(predicted_speeds, dist_from_segment_start, idx, 
                             waypoints, n_steps, waypoints_distances, DT):
    total_length = jnp.sum(waypoints_distances)
    s_relative = jnp.concatenate([
        jnp.array([dist_from_segment_start]),
        predicted_speeds * DT
    ]).cumsum()
    s_relative = s_relative % total_length  
    rolled_distances = jnp.roll(waypoints_distances, -idx)
    wp_dist_cum = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(rolled_distances)])
    index_relative = jnp.searchsorted(wp_dist_cum, s_relative, side='right') - 1
    index_relative = jnp.clip(index_relative, 0, len(rolled_distances) - 1)
    index_absolute = (idx + index_relative) % (waypoints.shape[0] - 1)
    next_index = (index_absolute + 1) % (waypoints.shape[0] - 1)
    seg_start = wp_dist_cum[index_relative]
    seg_len = rolled_distances[index_relative]
    t = (s_relative - seg_start) / seg_len
    p0 = waypoints[index_absolute][:, 1:3]
    p1 = waypoints[next_index][:, 1:3]
    interpolated_positions = p0 + (p1 - p0) * t[:, jnp.newaxis]
    s0 = waypoints[index_absolute][:, 0]
    s1 = waypoints[next_index][:, 0]
    interpolated_s = (s0 + (s1 - s0) * t) % waypoints[-1, 0]  
    yaw0 = waypoints[index_absolute][:, 3]
    yaw1 = waypoints[next_index][:, 3]
    interpolated_yaw = yaw0 + (yaw1 - yaw0) * t
    interpolated_yaw = (interpolated_yaw + jnp.pi) % (2 * jnp.pi) - jnp.pi
    v0 = waypoints[index_absolute][:, 5]
    v1 = waypoints[next_index][:, 5]
    interpolated_speed = v0 + (v1 - v0) * t
    reference = jnp.stack([
        interpolated_positions[:, 0],
        interpolated_positions[:, 1],
        interpolated_speed,
        interpolated_yaw,
        interpolated_s,
        jnp.zeros_like(interpolated_speed),
        jnp.zeros_like(interpolated_speed)
    ], axis=1)
    return reference

@jax.jit
def nearest_point_jax(point, trajectory, diffs):
    # diffs = trajectory[1:] - trajectory[:-1]                    
    l2s = jnp.sum(diffs**2, axis=1) + 1e-8                    
    dots = jnp.sum((point - trajectory[:-1]) * diffs, axis=1) 
    t = jnp.clip(dots / l2s, 0., 1.)   
    projections = trajectory[:-1] + diffs * t[:, None]
    dists = jnp.linalg.norm(point - projections, axis=1)      
    min_dist_segment = jnp.argmin(dists)                
    dist_from_segment_start = jnp.linalg.norm(diffs[min_dist_segment] * t[min_dist_segment])          
    return projections[min_dist_segment],dist_from_segment_start, dists[min_dist_segment], t[min_dist_segment], min_dist_segment

@njit(cache=True)
def nearest_point(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.
    Args:
        point (numpy.ndarray, (2, )): (x, y) of current pose
        trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
            NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the world
    Returns:
        nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
        nearest_dist (float): distance to the nearest point
        t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
        i (int): index of nearest point in the array of trajectory waypoints
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / (l2s + 1e-8)
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    projections = trajectory[:-1, :] + (t * diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    dist_from_segment_start = np.linalg.norm(diffs[min_dist_segment] * t[min_dist_segment])
    return projections[min_dist_segment], dist_from_segment_start, dists[min_dist_segment], t[
        min_dist_segment], min_dist_segment


# @njit(cache=True)
def get_reference_trajectory(predicted_speeds, dist_from_segment_start, idx, 
                             waypoints, n_steps, waypoints_distances, DT):
    s_relative = np.zeros((n_steps + 1,))
    s_relative[0] = dist_from_segment_start
    s_relative[1:] = predicted_speeds * DT
    s_relative = np.cumsum(s_relative)

    waypoints_distances_relative = np.cumsum(np.roll(waypoints_distances, -idx))

    index_relative = np.int_(np.ones((n_steps + 1,)))
    for i in range(n_steps + 1):
        index_relative[i] = (waypoints_distances_relative <= s_relative[i]).sum()
    index_absolute = np.mod(idx + index_relative, waypoints.shape[0] - 1)

    segment_part = s_relative - (
            waypoints_distances_relative[index_relative] - waypoints_distances[index_absolute])

    t = (segment_part / waypoints_distances[index_absolute])
    # print(np.all(np.logical_and((t < 1.0), (t > 0.0))))

    position_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, (1, 2)] -
                        waypoints[index_absolute][:, (1, 2)])
    position_diff_s = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, 0] -
                        waypoints[index_absolute][:, 0])
    orientation_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, 3] -
                            waypoints[index_absolute][:, 3])
    speed_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, 5] -
                    waypoints[index_absolute][:, 5])

    interpolated_positions = waypoints[index_absolute][:, (1, 2)] + (t * position_diffs.T).T
    interpolated_s = waypoints[index_absolute][:, 0] + (t * position_diff_s)
    interpolated_s[np.where(interpolated_s > waypoints[-1, 0])] -= waypoints[-1, 0]
    interpolated_orientations = waypoints[index_absolute][:, 3] + (t * orientation_diffs)
    interpolated_orientations = (interpolated_orientations + np.pi) % (2 * np.pi) - np.pi
    interpolated_speeds = waypoints[index_absolute][:, 5] + (t * speed_diffs)
    
    reference = np.array([
        # Sort reference trajectory so the order of reference match the order of the states
        interpolated_positions[:, 0],
        interpolated_positions[:, 1],
        interpolated_speeds,
        interpolated_orientations,
        # Fill zeros to the rest so number of references mathc number of states (x[k] - ref[k])
        interpolated_s,
        np.zeros(len(interpolated_speeds)),
        np.zeros(len(interpolated_speeds))
    ])
    return reference
    