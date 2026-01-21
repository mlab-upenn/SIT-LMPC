
"""An MPPI based planner."""
import os
import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import utilsuite
from utilsuite import jnumpify, jax_jit
from sit_lmpc.utils.plottings import LMPPI_Plottings

class LMPPI():
    """An MPPI based planner."""
    def __init__(self, config, env, jrng, track, logline,
                 temperature=0.01, damping=0.001):
        self.config = config
        self.n_iterations = config.n_iterations
        self.n_steps = config.n_steps
        self.n_samples = config.n_samples
        self.value_dim = config.value_dim
        self.a_std = jnp.array(config.control_sample_std)
        self.a_cov_shift = config.a_cov_shift
        self.adaptive_covariance = (config.adaptive_covariance and self.n_iterations > 1) or self.a_cov_shift
        self.accum_matrix = jnp.triu(jnp.ones((self.n_steps, self.n_steps)))
        self.a_shape = config.control_dim
        self.half_width = config.reduced_half_width
        self.temperature = config.temperature
        self.damping = config.damping
        self.s_frame_max = float(track.s_frame_max)
        self.zero_out_gamma = np.zeros((self.n_steps,)) ## only using the farthest point reward
        self.zero_out_gamma[-1] = 1
        self.logline = logline
        
        self.dp = utilsuite.DataProcessor()
        self.jrng = jrng
        self.plottings = LMPPI_Plottings(self.config)
        self.env = env
        self.env_state = None
        self.track = track
        self.nonequal_width = True if hasattr(self.track, 'tr_rights_jax') and self.track.tr_rights_jax else False
        
        lambs_sample_range = self.config.lambs_sample_range
        self.ss_lambs = jnp.arange(lambs_sample_range[0], 
                                   lambs_sample_range[1], 
                                   (lambs_sample_range[1]-lambs_sample_range[0])/self.config.n_lambs)
        self.boundary_lambs = jnp.arange(lambs_sample_range[0], 
                                   lambs_sample_range[1], 
                                   (lambs_sample_range[1]-lambs_sample_range[0])/self.config.n_lambs)
        self.sort_lambda(self.ss_lambs, self.boundary_lambs)
        self.init_state()
    
    def sort_lambda(self, ss_lambs, boundary_lambs):
        self.lambs = np.stack(np.meshgrid(ss_lambs, boundary_lambs), axis=-1).reshape(-1, 2)
        sort_inds = np.argsort(np.linalg.norm(self.lambs, axis=1))
        self.lambs = self.lambs[sort_inds]
        self.lambs = jnp.asarray(self.lambs)
        self.n_lambs = len(self.lambs)

    def update(self, env_state, params, value_fn, config, key_option, safe_set, state_f_0=None, state_c_0=None,
               dyna_norm_params=None, obstacle_list=[]):   

        self.config = config
        self.a_opt, self.a_cov = self.shift_prev_opt(self.a_opt, self.a_cov)        
        ret_num = self.lmppi_iteration_step(params, self.a_opt, self.a_cov, state_f_0, state_c_0, value_fn, env_state, self.jrng, 
                                                    safe_set, config.ss_relaxation, key_option,
                                                    dyna_norm_params, obstacle_list)
        # self.update_zt(numpify(self.traj_opt))
        return ret_num

    
    def lmppi_iteration_step(self, params, a_opt, a_cov, state_f_0, state_c_0, value_fn, env_state, jrng,
                             safe_set, ss_relaxation, key_option, dyna_norm_params=None, obstacle_list=[]):
        if self.config.state_predictor in self.config.cartesian_models:
            frenet_conversion_inds = self.track.get_s_search_range(state_f_0[0], self.config.frenet_conversion_track_range)
        else:
            frenet_conversion_inds = None
        
        states, actions, da, terminal_states, isnan, states_c = self.get_samples(a_opt, a_cov, params, jnp.asarray(env_state), jrng.new_key(), jrng.new_key(), 
                                                                         dyna_norm_params, state_f_0, frenet_conversion_inds)
        self.terminal_states_np = jnumpify(terminal_states)
        self.states_c = jnumpify(states_c)
        
        ## Get the value model prediction
        sampled_value = value_fn.get_value(value_fn.model.flax_train_state, 
                                              states, self.jrng.new_key(), value_fn.data_range)

        if len(obstacle_list) == 0:
            obstacle_dist = jnp.zeros((self.n_samples * self.n_steps))
        else:
            obstacle_dist = self.vmap_get_obstacle_distance(states, obstacle_list, self.config.obstacle_costfunc_size)

        # NaN error check
        if isnan:
            print('NaN error')
            return 1
        self.sampled_states = states
        self.sampled_actions = actions
        
        ## Find safe set hull
        if self.config.use_zt_ss_inds:
            self.ss_inds = safe_set.find_ss_inrange_zt(jnumpify(self.traj_opt[-1]).reshape(-1, 1), safe_set.ss_arr_frenet)
        else:
            self.ss_inds = safe_set.find_ss_inrange(self.terminal_states_np, safe_set.ss_arr_frenet, env_state)
        # print(len(self.ss_inds[0]))
        ref_ss_s = safe_set.ss_arr_frenet[self.ss_inds]
        if len(ref_ss_s) <= self.value_dim + 1:
            print('not enough safe set')
            return 1
        safe_set.update_convex_hull(ref_ss_s)

        ## Get the rewards
        r_sshull = safe_set.get_ss_dist(terminal_states)

        new_a_opt, da_opt, self.reward, self.r_value, self.r_sshull, self.r_boundary, r_action = \
            self.get_da_opt(r_sshull, sampled_value, states, da, a_opt, ss_relaxation, obstacle_dist)
        # if isnan: 
        #     self.logline('new_a_opt NaN error')
        #     return (1,) * 13
        
        ## Rollout state predictor
        # if self.config.state_predictor in ['nn_dynamic_ST']:
        #     traj_opts = self.rollout(new_a_opt, 
        #                             jnp.repeat(env_state[None, :], self.config.n_lambs**2, axis=0),
        #                             self.jrng.new_key())
        # else:
        if len(obstacle_list) == 0:
            traj_opts, boundary_distance, traj_opts_terminals, traj_opts_c, boundary_distance_trajs = self.get_traj_opts(new_a_opt, env_state, params, self.jrng.new_key(), 
                                                                                dyna_norm_params, state_f_0, frenet_conversion_inds)
        else:
            traj_opts, boundary_distance, traj_opts_terminals, traj_opts_c, boundary_distance_trajs = self.get_traj_opts_w_obs(new_a_opt, env_state, params, self.jrng.new_key(), 
                                                                                dyna_norm_params, state_f_0, frenet_conversion_inds, obstacle_list)            
        
        # if self.config.state_predictor in self.config.cartesian_models:
        #     traj_opts = self.convert_cartesian_to_frenet_jax(traj_opts, state_f_0[0])

        ## Check constraints
        s_opt_ss_distance = safe_set.get_ss_dist(traj_opts_terminals)      
        # self.a_opt, a_opt_cpu, self.traj_opt, traj_opt_c, = self.get_lamb_ind(boundary_distance, s_opt_ss_distance, new_a_opt, traj_opts, ss_relaxation, traj_opts_c)
        self.a_opt, self.traj_opt, self.traj_opt_c, \
        self.lamb_ind, self.lambs_opt, self.s_opt_ss_distance, self.boundary_distance, violation = self.get_lamb_ind(boundary_distance, s_opt_ss_distance, new_a_opt, traj_opts, ss_relaxation, traj_opts_c)

        self.boundary_distance = (self.boundary_distance - self.config.reduced_half_width) * (self.boundary_distance > self.config.reduced_half_width)
        # self.a_opt = numpify(self.a_opt)[self.lamb_ind]
        
        if key_option == '1' and self.config.plottings:
        # if self.boundary_distance > 0:
            states = states.reshape(self.n_samples, self.n_steps, states.shape[-1])
            self.plottings.plot_rewards_and_sample(0, ref_ss_s, states, 
                    traj_opts, self.lamb_ind, self.lambs_opt, self.r_value, boundary_distance_trajs, self.reward, self.r_sshull, self.r_boundary, self.config.reduced_half_width, 
                    self.s_opt_ss_distance, self.boundary_distance, actions, a_opt, a_opt, show=True)
        return 0

    # @partial(jax.jit, static_argnums=(0))
    @jax_jit(static_argnums=(0,))
    def get_lamb_ind(self, boundary_distance, s_opt_ss_distance, new_a_opt, traj_opts, ss_relaxation, traj_opts_c):
        s_opt_ss_distance = s_opt_ss_distance * (s_opt_ss_distance > ss_relaxation)
        violation = boundary_distance * 1e3 + s_opt_ss_distance
        lamb_ind = jnp.argmin(violation)
        lambs_opt = self.lambs[lamb_ind]
        a_opt = new_a_opt[lamb_ind]
        traj_opt = traj_opts[lamb_ind]
        traj_opt_c = traj_opts_c[lamb_ind]
        s_opt_ss_distance = s_opt_ss_distance[lamb_ind]
        boundary_distance = boundary_distance[lamb_ind]
        return a_opt, traj_opt, traj_opt_c, lamb_ind, lambs_opt, s_opt_ss_distance, boundary_distance, violation

    # @partial(jax.jit, static_argnums=(0))
    @jax_jit(static_argnums=(0,))
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

    def init_state(self):
        a_shape = jnp.prod(self.a_shape)  # np.int32
        self.a_opt = jnp.zeros((self.n_steps, a_shape))
        # a_cov: [n_steps, a_shape, a_shape]
        self.a_cov = (self.a_std**2)*jnp.tile(jnp.eye(a_shape), (self.n_steps, 1, 1))
        self.a_cov_init = self.a_cov.copy()
        self.sampled_states = jnp.zeros((self.n_samples, self.n_steps, self.config.state_dim))
        self.s_opt_ss_distance, self.boundary_distance, self.ss_inds, self.lambs_opt = [0], [0], 0, [0, 0]
        self.traj_opt = jnp.zeros((self.n_steps, self.value_dim))
    
    @jax_jit(static_argnums=(0,))
    def get_samples(self, a_opt, a_cov, params, env_state, rng_key, rng_key2, dyna_norm_params, state_f_0, frenet_conversion_inds):        
        da = utilsuite.truncated_gaussian_sampler(rng_key, 0., -jnp.ones_like(a_opt) - a_opt, jnp.ones_like(a_opt) - a_opt, 
                                        shape=(self.n_samples, self.n_steps, 2),
                                        std=jnp.array(self.config.control_sample_std))
        # jax.debug.print('dyna_norm_params {x}', x=dyna_norm_params.shape)
        actions = jnp.clip(jnp.expand_dims(a_opt, axis=0) + da, -1.0, 1.0)
        states, env_vars = jax.vmap(self.rollout, in_axes=(0, None, None, None, None))(
            actions, env_state, params, rng_key2, dyna_norm_params
        )
        
        if self.config.state_predictor in self.config.cartesian_models:
            states_c = states.copy()[:, -1, :self.config.value_dim]
            states = self.convert_cartesian_to_frenet(states, frenet_conversion_inds)
        else:
            states_c = None
        
        ## check if the sampled states are crossing the finish line
        states_s = states.reshape(-1, states.shape[-1])[:, 0]
        states_and_state = jnp.concatenate([states_s, state_f_0[None, 0]], axis=0)
        center_s = (jnp.max(states_and_state) + jnp.min(states_and_state))/2
        cross_inds = (states_s - center_s) < (-self.s_frame_max/3)
        states_s = states_s + cross_inds * self.s_frame_max
        states = jnp.concatenate([states_s.reshape(self.n_samples, self.n_steps, 1), states[:, :, 1:]], axis=2)
        states_unstack = states.reshape(-1, states.shape[-1])
        
        return states_unstack, actions, da, states[:, -1, :self.config.value_dim], jnp.isnan(states).any(), states_c
    
    def check_crossing(self, states, s_frame_max):
        """
        check if the sampled states are crossing the boundary, by seeing if the sampled states have two split far parts
        """
        center_s = (np.max(states) + np.min(states))/2
        cross_inds = np.where((states - center_s) < -s_frame_max/3)
        return cross_inds, center_s
    
    @partial(jax.jit, static_argnums=(0))
    def get_reward_func(self, lambs, r_value, r_sshull, r_boundary, r_action):
        reward = r_value - r_sshull * lambs[0] - r_boundary * lambs[1] - r_action
        return reward
    
    @partial(jax.jit, static_argnums=(0))
    def get_da_opt(self, r_sshull, r_value, sampled_states, da, a_opt, 
                   ss_relax_radius, obstacle_dist):
        # track_left, track_right = self.track.vmap_get_track_width(sampled_states)
        # track_right = track_right - self.config.half_width_costfunc_buffer
        # track_left = track_left - self.config.half_width_costfunc_buffer
        # boundary_out_left = (sampled_states[:, 1] < -track_left) * (sampled_states[:, 1] < 0)
        # boundary_out_right = (sampled_states[:, 1] > track_right) * (sampled_states[:, 1] >= 0)
        # r_boundary = (jnp.abs(sampled_states[:, 1]) - track_right) * boundary_out_right + \
        #                 (jnp.abs(sampled_states[:, 1]) - track_left) * boundary_out_left
        
        sampled_states = sampled_states.reshape(self.n_samples, self.n_steps, self.config.state_dim)
        boundary_out = jnp.abs(sampled_states[:, :, 1]) > self.config.reduced_half_width
        # r_boundary = jnp.abs(sampled_states[:, :, 1]) - self.config.half_width
        r_boundary = jnp.abs(sampled_states[:, :, 1]) * boundary_out
        # r_boundary = r_boundary + obstacle_dist
        # r_boundary = r_boundary + (r_boundary > 0)
        r_boundary = r_boundary - jnp.min(r_boundary)
        r_boundary = jax.lax.select(jnp.max(r_boundary) != 0, r_boundary / jnp.max(r_boundary), r_boundary)
        r_boundary = r_boundary.reshape(self.n_samples, self.n_steps)
        
        r_value = r_value.reshape(self.n_samples, self.n_steps)
        if self.config.take_mean_of_values: r_value = jnp.mean(r_value, axis=1, keepdims=True)
        r_value = r_value - jnp.min(r_value)
        r_value = jax.lax.select(jnp.max(r_value) != 0, r_value / jnp.max(r_value), r_value)
        r_value = r_value * self.zero_out_gamma
        
        r_sshull = (r_sshull - ss_relax_radius) * (r_sshull > ss_relax_radius)
        r_sshull = r_sshull[:, None]
        r_sshull = r_sshull * (r_sshull < self.config.r_sshull_limit) + \
            self.config.r_sshull_limit * (r_sshull >= self.config.r_sshull_limit) # normalize r_sshull
        r_sshull = r_sshull - jnp.min(r_sshull)
        r_sshull = jax.lax.select(jnp.max(r_sshull) != 0, r_sshull / jnp.max(r_sshull), r_sshull)
        r_sshull = r_sshull * self.zero_out_gamma # only use the terminal state reward
        
        r_action = jnp.zeros((self.n_samples, 1))
        r_action = jnp.concatenate([r_action, jnp.linalg.norm(da[:, 1:, :] + a_opt[1:, :] - da[:, :-1, :] - a_opt[:-1, :], axis=-1)], axis=-1)
        r_action = r_action * self.config.action_rate_penalty
        
        reward = jax.vmap(self.get_reward_func, in_axes=(0, None, None, None, None))(self.lambs, r_value, r_sshull, r_boundary, r_action)
        R = jax.vmap(jax.vmap(self.returns))(reward)  # [n_lambs, n_samples, n_steps], pylint: disable=invalid-name
        weights = jax.vmap(jax.vmap(self.normalized_weights, 1, 1))(R)  # [n_lambs, n_samples, n_steps]
        da_opt = jax.vmap(jax.vmap(jnp.average, (1, None, 1)), in_axes=(None, None, 0))(da, 0, weights)
        new_a_opt = jnp.clip(da_opt + a_opt, -1.0, 1.0)
        return new_a_opt, da_opt, reward, r_value, r_sshull, r_boundary, r_action
        # return new_a_opt, da_opt, reward, r_value, r_sshull, r_boundary, r_action, jnp.isnan(new_a_opt).any()
    
    @partial(jax.jit, static_argnums=(0))
    def vmap_get_obstacle_distance(self, states, obstacle_list, obstacle_size):
        return jax.vmap(self.get_obstacle_distance, in_axes=[0, None, None])(states, obstacle_list, obstacle_size)  
    
    @partial(jax.jit, static_argnums=(0))
    def get_obstacle_distance(self, state, obstacle_list, obstacle_size):
        dist = np.min(jnp.linalg.norm(jnp.array(obstacle_list)[:, :2] - state[:2], axis=1), axis=0)
        return jax.lax.select(dist < obstacle_size, (obstacle_size - dist) + 1, 0.)
    
    @partial(jax.jit, static_argnums=(0))
    def get_traj_opts_w_obs(self, new_a_opt, env_state, params, rng_key, dyna_norm_params, state_f_0, frenet_conversion_inds, obstacle_list):
        traj_opts, env_vars = jax.vmap(self.rollout, in_axes=(0, None, None, None, None))(
            new_a_opt, env_state, params, rng_key, dyna_norm_params
        )
        traj_opts_c = traj_opts.copy()
        traj_opts = traj_opts.reshape(-1, traj_opts.shape[-1])
        if self.config.state_predictor in self.config.cartesian_models:
            traj_opts = self.convert_cartesian_to_frenet(traj_opts, frenet_conversion_inds)
        obstacle_dist = self.vmap_get_obstacle_distance(traj_opts, obstacle_list, self.config.obstacle_size).reshape(self.n_lambs, self.n_steps)

        ## check if the sampled states are crossing the finish line
        traj_opts_s = traj_opts[:, 0]
        traj_opts_and_state = jnp.concatenate([traj_opts_s, state_f_0[None, 0]], axis=0)
        center_s = (jnp.max(traj_opts_and_state) + jnp.min(traj_opts_and_state))/2
        cross_inds = (traj_opts_s - center_s) < (-self.s_frame_max/3)
        traj_opts_s = traj_opts_s + cross_inds * self.s_frame_max
        traj_opts = jnp.concatenate([traj_opts_s.reshape(self.n_lambs * self.n_steps, 1), traj_opts[:, 1:]], axis=1)

        if self.nonequal_width:
            track_left, track_right = self.track.vmap_get_track_width(traj_opts)
            track_right = track_right - self.config.half_width_buffer
            track_left = track_left - self.config.half_width_buffer
            boundary_out_left = (traj_opts[:, 1] < -track_left) * (traj_opts[:, 1] < 0)
            boundary_out_right = (traj_opts[:, 1] > track_right) * (traj_opts[:, 1] >= 0)
            boundary_distance = (jnp.abs(traj_opts[:, 1]) - track_right) * boundary_out_right + \
                            (jnp.abs(traj_opts[:, 1]) - track_left) * boundary_out_left
        else:
            boundary_distance = jnp.abs(traj_opts[:, 1])
            boundary_distance = boundary_distance * (boundary_distance > self.config.reduced_half_width)
        traj_opts = traj_opts.reshape(self.n_lambs, self.n_steps, traj_opts.shape[-1])
        boundary_distance_trajs = boundary_distance.reshape(self.n_lambs, self.n_steps) + obstacle_dist
        boundary_distance = jnp.sum(boundary_distance_trajs, axis=1)
        return traj_opts, boundary_distance, traj_opts[:, -1, :self.config.value_dim], traj_opts_c, boundary_distance_trajs
    
    @partial(jax.jit, static_argnums=(0))
    def get_traj_opts(self, new_a_opt, env_state, params, rng_key, dyna_norm_params, state_f_0, frenet_conversion_inds):
        traj_opts, env_vars = jax.vmap(self.rollout, in_axes=(0, None, None, None, None))(
            new_a_opt, env_state, params, rng_key, dyna_norm_params
        )
        traj_opts_c = traj_opts.copy()
        traj_opts = traj_opts.reshape(-1, traj_opts.shape[-1])
        if self.config.state_predictor in self.config.cartesian_models:
            traj_opts = self.convert_cartesian_to_frenet(traj_opts, frenet_conversion_inds)
        
        ## check if the sampled states are crossing the finish line
        traj_opts_s = traj_opts[:, 0]
        traj_opts_and_state = jnp.concatenate([traj_opts_s, state_f_0[None, 0]], axis=0)
        center_s = (jnp.max(traj_opts_and_state) + jnp.min(traj_opts_and_state))/2
        cross_inds = (traj_opts_s - center_s) < (-self.s_frame_max/3)
        traj_opts_s = traj_opts_s + cross_inds * self.s_frame_max
        traj_opts = jnp.concatenate([traj_opts_s.reshape(self.n_lambs * self.n_steps, 1), traj_opts[:, 1:]], axis=1)
        
        if self.nonequal_width:
            track_left, track_right = self.track.vmap_get_track_width(traj_opts)
            track_right = track_right - self.config.half_width_costfunc_buffer
            track_left = track_left - self.config.half_width_costfunc_buffer
            boundary_out_left = (traj_opts[:, 1] < -track_left) * (traj_opts[:, 1] < 0)
            boundary_out_right = (traj_opts[:, 1] > track_right) * (traj_opts[:, 1] >= 0)
            boundary_distance = (jnp.abs(traj_opts[:, 1]) - track_right) * boundary_out_right + \
                            (jnp.abs(traj_opts[:, 1]) - track_left) * boundary_out_left
        else:
            boundary_distance = jnp.abs(traj_opts[:, 1])
            boundary_distance = boundary_distance * (boundary_distance > self.config.reduced_half_width)
        boundary_distance_trajs = boundary_distance.reshape(self.n_lambs, self.n_steps)
        boundary_distance = jnp.sum(boundary_distance_trajs, axis=1) 
        traj_opts = traj_opts.reshape(self.n_lambs, self.n_steps, traj_opts.shape[-1])
        return traj_opts, boundary_distance, traj_opts[:, -1, :self.config.value_dim], traj_opts_c, boundary_distance_trajs

    @partial(jax.jit, static_argnums=(0))
    def returns(self, r): # r: [n_steps]
        return jnp.dot(self.accum_matrix, r)  # R: [n_steps]
    
    @partial(jax.jit, static_argnums=(0))
    def normalized_weights(self, R):  # pylint: disable=invalid-name
        R_stdzd = (R - jnp.max(R)) / ((jnp.max(R) - jnp.min(R)) + self.damping)  # pylint: disable=invalid-name
        w = jnp.exp(R_stdzd / self.temperature)  # [n_samples] np.float32
        w = w / jnp.sum(w)  # [n_samples] np.float32
        return w
    
    @jax_jit(static_argnums=(0,))
    def rollout(self, actions, env_state, params, rng_key, norm_params):
        """
        # actions: [n_steps, a_shape]
        # env: {.step(states, actions)}
        # env_state: np.float32
        # actions: # a_0, ..., a_{n_steps}. [n_steps, a_shape]
        # states: # s_1, ..., s_{n_steps+1}. [n_steps, env_state_shape]
        """
        
        def rollout_step(env_state, actions):
            (env_state, env_var, mb_dyna) = self.env.step(env_state, actions, params, rng_key, norm_params)
            return env_state, env_var

        states = []
        env_vars = []
        for t in range(self.n_steps):
            env_state, env_var = rollout_step(env_state, actions[t, :])
            # env_state = rollout_step(env_state, actions[:, t, :])
            states.append(env_state)
            env_vars.append(env_var)
        return jnp.asarray(states), jnp.asarray(env_vars)
        # return jnp.asarray(states).transpose(1, 0, 2)  # [n_samples, n_steps, env_state_shape]
        
    @partial(jax.jit, static_argnums=(0))
    def weight_a_cov(self, w, da):
        a_cov = jax.vmap(jax.vmap(jnp.outer))(
            da, da
        )  # [n_samples, n_steps, a_shape, a_shape]
        a_cov = jax.vmap(jnp.average, (1, None, 1))(
            a_cov, 0, w
        )  # a_cov: [n_steps, a_shape, a_shape]
        a_cov = a_cov + jnp.eye(self.a_shape)*0.00001 # prevent loss of rank when one sample is heavily weighted
        return a_cov
    
    @partial(jax.jit, static_argnums=(0))
    def convert_cartesian_to_frenet(self, states, s_ref_inds):
        states_shape = (*states.shape[:-1], 7)
        states = states.reshape(-1, states.shape[-1])
        converted_states = self.track.vmap_cartesian_to_frenet_jax_jit(states[:, (0, 1, 4)], s_ref_inds)
        states_frenet = jnp.concatenate([converted_states[:, :2], 
                                         states[:, 2:4],
                                         converted_states[:, 2:3],
                                         states[:, 5:7]], axis=-1)
        # states_frenet = jnp.concatenate([converted_states[:, :2], 
        #                                  states[:, 2:3],
        #                                  states[:, 3:4] * jnp.cos(states[:, 6:7]),
        #                                  converted_states[:, 2:3],
        #                                  states[:, 5:6],
        #                                  states[:, 3:4] * jnp.sin(states[:, 6:7])], axis=-1)
        return states_frenet.reshape(states_shape)

