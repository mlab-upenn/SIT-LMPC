import os
import pathlib
import sys
import time
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import utilsuite

import sit_lmpc
from f1tenth_gym.envs.f110_env import F110Env
from sit_lmpc.planners.lmppi.infer_env import InferEnv
from sit_lmpc.planners.lmppi.mppi_tracking import MPPI
from sit_lmpc.utils.mppi_renderers import MPPIRenderers
from sit_lmpc.utils.track_jax import Track

np.set_printoptions(suppress=True, precision=10)
jax.config.update("jax_compilation_cache_dir", str(pathlib.Path("~/jax_cache").expanduser()))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.01)


## This is a demosntration of how to use the MPPI planner with the new F1Tenth Gym environment
## Zirui Zang 2025/06/25

class Config(utilsuite.ConfigYAML):
    exp_name = 'mppi_tracking'
    sim_time_step = 0.1
    render = 1
    kmonitor_enable = 0
    save_dir = str(Path(sit_lmpc.__file__).parent) + '/results/' + exp_name + '/'
    max_lap = 300
    random_seed = None
    integrator_timestep = 0.02
    
    use_blank_map = True
    map_scale = 1
    friction = 1
    init_vel = 10
    map_ind = 0
    map_names = {0: str(Path(sit_lmpc.__file__).parent) + '/maps/custom1/custom1',
                 1: str(Path(sit_lmpc.__file__).parent) + '/maps/Spielberg/Spielberg'}
    n_steps = 10
    n_samples = 512
    n_iterations = 1
    control_dim = 2
    control_sample_std = [1.0, 1.0]
    state_predictor = 'dynamic_ST' # ['pacjeka_frenet', 'ks_frenet', 'dynamic_ST', 'kinematic_ST', 'nn_dynamic_ST']
    gym_model = 'st' # ['st', 'ks', 'mb']
    cartesian_models = ['dynamic_ST', 'kinematic_ST', 'nn_dynamic_ST']
    half_width = 4
    a_cov_shift = 0
    norm_params = [[0.8, 7.0], [-0.4, -3.5]] # [[range of steer_vel, range of accel], [min of steer_vel, min of accel]]

    adaptive_covariance = False
    added_noise = [5e-3, 5e-3, 5e-3] # control_vel, control_steering, state 

@jax.jit
def jit_concatenate(sampled_states):
    return jax.device_get(jnp.concatenate(sampled_states[:, :, (0, 1, 4)]))[:, :2]

@jax.jit
def jit_device_get(array):
    return jax.device_get(array)

def main():
    
    config = Config()
    config.norm_params = np.array(config.norm_params)
    
    if config.random_seed is None:
        config.random_seed = np.random.randint(0, 1e6)
    jrng = utilsuite.oneLineJaxRNG(config.random_seed)   
    print('random seed', config.random_seed)
    
    us, logline = utilsuite.utilitySuite(config)
    map_path = pathlib.Path(config.map_names[config.map_ind]).resolve()
    dynamic_params = F110Env.fullscale_vehicle_params()
    gym_config = {
        "send": config.random_seed,
        "map": str(map_path),
        "timestep": 0.01,
        "integrator_timestep": 0.01,
        "control_input": ["accl", "steering_speed"],
        "model": config.gym_model,
        "observation_config": {"type": "direct"},
        "params": F110Env.fullscale_vehicle_params(),
        "map_scale": 1.0,
        "enable_rendering": config.render,
        "enable_scan": 0,
        "compute_frenet": 1,
        "max_laps": 5,
    }
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=gym_config,
        render_mode="unlimited",
    )
    additional_render = 1
    track = Track.from_track_path(map_path)
        
    if config.render:
        mppi_renderers = MPPIRenderers(env.unwrapped.renderer, track, config)
        mppi_renderers.set_track_renderer(max_points=200, boundary_color=us.colorpal.rgb('o'), centerline_color=(50, 50, 50), z_offset=0.01)
        ref_renderer = mppi_renderers.get_point_renderer(np.array([0, 0]), color=us.colorpal.rgb('pi'), size=10, z_offset=0.02)
        sampled_renderer = mppi_renderers.get_point_renderer(np.array([0, 0]), color=us.colorpal.rgb('g'), size=5, z_offset=0.02)
        opt_traj_renderer = mppi_renderers.get_point_renderer(np.array([0, 0]), color=us.colorpal.rgb('y'), size=10, z_offset=0.04)

    infer_env = InferEnv(track, config, DT=config.sim_time_step, params_dict=dynamic_params)
    mppi = MPPI(config, infer_env, jrng, track.raceline.waypoints)      
    frenet_start = np.array(env.unwrapped.track.frenet_to_cartesian(0.0, 0, 0))
    init_state = np.array([[frenet_start[0], frenet_start[1], 0, config.init_vel, 0, 0, 0]])
    obs, info = env.reset(options={'states':init_state})
    
    done = False
    if config.render:
        env.render()
    
    while not done:
        state_c_0 = obs['agent_0']["std_state"]
        state_c_0 += np.random.normal(scale=config.added_noise[2], size=state_c_0.shape)
        us.timer.tic('step', 500)
        
        ## MPPI call
        reference_traj = mppi.get_refernece_traj_jax(state_c_0.copy(), target_speed=config.init_vel, n_steps=config.n_steps)   
        mppi.update(infer_env.params, jnp.array(state_c_0), jnp.array(reference_traj))
        if config.render:
            ref_renderer.update(reference_traj[:, :2])
            opt_traj_renderer.update(jit_device_get(mppi.traj_opt[:, :2]))
            sampled_renderer.update(jit_concatenate(mppi.sampled_states))
        
            
        ## Gym call 
        control = utilsuite.jnumpify(mppi.a_opt[0]) * config.norm_params[0, :2]/2
        control = [[control[0] + np.random.normal(scale=config.added_noise[0]), 
                   control[1] + np.random.normal(scale=config.added_noise[1])]]
        obs, step_reward, done, truncated, info = env.step(np.array(control))

        if config.render:
            env.render()
            
        us.timer.toc('step', Hz=True)
        
            
if __name__ == '__main__':
    main()
    
