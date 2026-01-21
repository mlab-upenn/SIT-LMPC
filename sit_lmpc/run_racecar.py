import os, sys, time, copy, pathlib, threading
import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym

np.set_printoptions(suppress=True, precision=10)
jax.config.update("jax_compilation_cache_dir", str(pathlib.Path("~/jax_cache").expanduser()))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.01)
os.environ["DEBUG_LEVEL_ZZR"] = "0"

import sit_lmpc
from sit_lmpc.planners.lmppi.lmppi import LMPPI
from sit_lmpc.planners.lmppi.infer_env import InferEnv
from sit_lmpc.planners.lmppi.safe_set import SafeSet
from sit_lmpc.planners.lmppi.value_fn import ValueFn
from sit_lmpc.planners.lmppi.mppi_tracking import MPPI
import utilsuite
from sit_lmpc.utils.plottings import Plottings
from sit_lmpc.utils.track_jax import Track
from sit_lmpc.utils.mppi_renderers import MPPIRenderers
from sit_lmpc.learned_dyna_model import dynaConfig, get_dyna_train_data
from f1tenth_gym.envs.f110_env import F110Env

def main():

    test_times = 1
    init_run_id = 0
    min_laptimes = []
    max_laps = []
    experiment_maps = [1]
    for map_ind in experiment_maps:
        test_ind = init_run_id
        while test_ind < test_times + init_run_id:
            
            config = utilsuite.ConfigYAML()
            curr_dir = str(pathlib.Path(sit_lmpc.__file__).parent.resolve())
            config.curr_dir = curr_dir
            config.load(curr_dir + '/config/exp_config.yaml')
            config.norm_params = np.array(config.norm_params)
            value_config = utilsuite.ConfigYAML()
            value_config.load(curr_dir + '/config/value_config.yaml')

            exp_name = 'f1_'
            config.run = test_ind
            config.freeze_render_when_stop = 0
            config.kmonitor_enable = 0
            config.plottings = 1
            config.quick_load = 0
            config.quick_load_init = 0
            config.ss_hull_precompile = 1
            config.map_ind = map_ind
            config.load_dir = curr_dir + '/results/' + exp_name + f'map{config.map_ind}' + '/'
            config.insight_len = 40
            
            config.state_predictor = 'dynamic_ST' # ['dynamic_ST', 'kinematic_ST', 'nn_dynamic_ST']
            config.gym_model = 'st' # "ks", "st", "mb"
            config.cartesian_models = ['dynamic_ST', 'kinematic_ST', 'nn_dynamic_ST']
            config.load_pretrained_dynamic_model = 1
            config.friction = 1.0
            config.init_vel = 10
            config.n_steps = 10
            config.control_sample_std = [0.25, 0.25]
            config.random_seed = np.random.randint(0, 1e6)
            value_config.random_seed = config.random_seed
            
            config.half_width = 4.
            config.half_width_buffer = 0.03
            config.reduced_half_width = config.half_width - config.half_width_buffer
            config.n_lambs = 16
            config.lambs_sample_range = [0, 20]
            config.ss_relaxation = 0.03
            
            exp_name = exp_name + f'r{config.run}_' + config.state_predictor + f'_map{config.map_ind}' + f'_steps{config.n_steps}'
            config.exp_name = exp_name
            
            config.save_dir = config.load_dir + exp_name + '/'    
            config.save(config.save_dir + 'config.yaml')
            value_config.save(config.save_dir + 'value_config.yaml')
            if config.state_predictor in ['nn_dynamic_ST']:
                config.train_dyna_model = 1
                dyna_config = dynaConfig()
                dyna_config.load(dyna_config.datadir + 'config.yaml')
                dyna_config.random_seed = config.random_seed
                dyna_config.save(config.save_dir + 'dyna_config.yaml')
                dyna_config.silent = 1
                dyna_config.dyna_norm_params = jnp.array(dyna_config.dyna_norm_params)
            else:
                dyna_config = dynaConfig()
                dyna_config.dyna_norm_params = None
            
            ret_num, us = run_experiment(config, value_config, dyna_config)
            us.logline('test_ind', test_ind, 'ret_num', ret_num)
            us.logline('')
            if ret_num != 1:
                test_ind += 1
                if len(us.rec.laptime_record) > 0:
                    min_laptimes.append(np.min(us.rec.laptime_record))
                    max_laps.append(len(us.rec.laptime_record))
                else:
                    min_laptimes.append(0)
                    max_laps.append(0)
                us.logline('min_laptimes', min_laptimes)
                us.logline('max_laps', max_laps)
        us.logline('experiment done', exp_name)
        us.logline('')
    

def run_experiment(config, value_config, dyna_config):
    us, logline = utilsuite.utilitySuite(config)
    logline('exp_name', config.exp_name)
    logline('random seed', config.random_seed)
    np.random.seed(config.random_seed)
    jrng = utilsuite.oneLineJaxRNG(config.random_seed)
    
    map_path = config.curr_dir + config.map_inds[config.map_ind]
    dynamic_params = F110Env.fullscale_vehicle_params()
    dynamic_params['length'] = dynamic_params['width']
    dynamic_params['mu'] = config.friction
    gym_config = {
        "send": config.random_seed,
        "map": str(map_path),
        "timestep": 0.1,
        "integrator_timestep": 0.02,
        "control_input": ["accl", "steering_speed"],
        "model": config.gym_model, 
        "observation_config": {"type": "direct"},
        "params": dynamic_params,
        "map_scale": 1.0,
        "enable_rendering": config.render,
        "enable_scan": 0,
        "compute_frenet": 1,
        "max_laps": 'inf',
        "steer_delay_buffer_size": 0
    }
    
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=gym_config,
        # render_mode="human",
        render_mode="unlimited",
    )
    track = Track.from_track_path(map_path)
    frenet_start = np.array(env.unwrapped.track.frenet_to_cartesian(0.0, 0, 0))
    init_state = np.array([[frenet_start[0], frenet_start[1], 0, config.init_vel, 0, 0, 0]])
    obs, info = env.reset(options={'states':init_state})   
    
    plottings = Plottings(track, config, us, enable=config.plottings)
    infer_env = InferEnv(track, config, DT=config.sim_time_step, params_dict=dynamic_params, dyna_config=dyna_config)
    mppi = MPPI(config, infer_env, jrng, track.raceline.waypoints)
    safe_set = SafeSet(config, logline, track)
    lmppi = LMPPI(config, infer_env, jrng, track, logline)
    if config.ss_hull_precompile: safe_set.ss_hull_precompile(lmppi.lambs.shape[0])
    value_fn = ValueFn(us, value_config, config)
    
    laptime = 0.0
    lap_cnt = 0 
    start_time = time.time()
    us.rec.init('laptimes', # record per action laptime
                's_states', # record in-loop frenet poses
                'xy_states', # record in-loop cartesian poses
                'safe_set', 'safe_set_xy', 'dyna_train_data',
                'control_record', 'control_records', 
                'a_opt_record', 'a_opt_records', 'mean_vels', 'traj_opt',
                'ss_violation_record', 'ss_average_record', 'ss_max_record', 
                'boundary_violation_record', 'boundary_average_record', 'boundary_max_record', 
                'laptime_record', 'lamb_record', 'lamb_records', 'mean_lambs',
                'min_laptime', 'min_states', 'time_record')
    us.rec.min_laptime = np.inf
    
    # Quick load
    if config.quick_load:
        us.rec.load_onefile(save_dir=config.load_dir, filename='init')
        logline('quick load from', config.load_dir)
        safe_set.safe_set_frenet = copy.deepcopy(us.rec.safe_set)
        safe_set.safe_set_carti = copy.deepcopy(us.rec.safe_set_xy)
        safe_set.update_ss_arr()                                           
        
        if config.quick_load_init:
            lmppi.a_opt = us.rec.a_opt_records[-1][-1][:config.n_steps]
            config.init_vel = float(us.rec.init_state[3])
            obs, step_reward, done, info = env.reset(options={'states': np.array([us.rec.init_state])})
        logline('init_state', np.asarray(obs['agent_0']['state']))
        
        value_fn.thread = threading.Thread(target=value_fn.train_valuefn, args=(safe_set.safe_set_frenet, config, 
                                                                                value_fn.thread_ret, value_config.max_epoch*2))
        value_fn.thread.start()
        
        # if config.train_dyna_model and dyna_config is not None:
            # dyna_train_data_train = us.rec.dyna_train_data
            # control_records_train = us.rec.control_records
        #     data_in, data_out, infer_env.dyna_config = get_dyna_train_data(us.dp, dyna_train_data_train, control_records_train, infer_env.dyna_config)
        #     infer_env.dyna_model.flax_train_state, dynamic_model_losses = infer_env.dyna_model.train(infer_env.dyna_model.flax_train_state,
        #                                                                     data_out, data_in, max_epoch=dyna_config.max_epoch, loss_threshold=0.001)
        #     logline('dynamic_model_losses', dynamic_model_losses[-1], print_line=config.print_line)
        value_fn.thread.join()
        # value_fn.load_valuefn(safe_set.safe_set_frenet, config)
        value_fn.thread = None
        
    if config.render:
        mppi_renderers = MPPIRenderers(env.unwrapped.renderer, track, config)
        mppi_renderers.set_track_renderer(max_points=200, boundary_color=us.colorpal.rgb('o'), centerline_color=(50, 50, 50), z_offset=0.01)
        ref_renderer = mppi_renderers.get_point_renderer(np.array([0, 0]), color=us.colorpal.rgb('pi'), size=10, z_offset=0.02)
        # sampled_renderer = mppi_renderers.get_point_renderer(np.array([0, 0]), color=us.colorpal.rgb('g'), size=5, z_offset=0.02)
        opt_traj_renderer = mppi_renderers.get_point_renderer(np.array([0, 0]), color=us.colorpal.rgb('y'), size=10, z_offset=0.04)
        

    
    LOOP = True
    pbar = utilsuite.coloredTqdm(config.max_lap, color='g')
    while LOOP:
        initial_loops = len(us.rec.safe_set) < config.ss_size
        key_option = us.kmonitor.option()
        state_c_0 = obs['agent_0']['std_state'].copy()
        if state_c_0[3] < 0.1: # return error when speed is too low
            logline('speed too low')
            return end_action(2, us, config, plottings)
        state_f_0 = state_c_0.copy()
        state_f_0[[0, 1, 4]] = obs['agent_0']['frenet_pose'].copy()
        # print('state_f_0', state_f_0)
        # state_f_0[3] = state_c_0[3] * np.cos(state_c_0[6]) # update velocity in frenet frame
        
        env_state = state_c_0.copy()
        ## MPPI call
        if initial_loops:
            reference_traj = mppi.get_refernece_traj_jax(state_c_0.copy(), target_speed=config.init_vel, n_steps=config.n_steps) 
            mppi.update(infer_env.params, jnp.array(state_c_0), jnp.array(reference_traj), dyna_norm_params=dyna_config.dyna_norm_params)
            control = utilsuite.jnumpify(mppi.a_opt[0]) * config.norm_params[0, :2]/2
        else:
            
            ret_num = lmppi.update(jnp.asarray(env_state + 
                                                get_noise_truncated_normal(config.added_noise[2], config.added_noise_limit_multiplier, env_state.shape)), 
                                    infer_env.params, value_fn, config, key_option, safe_set, state_f_0=state_f_0, state_c_0=state_c_0, dyna_norm_params=dyna_config.dyna_norm_params)
            reference_traj = safe_set.ss_arr_carti[:, :5][lmppi.ss_inds] # for rendering
            us.rec.traj_opt.append(lmppi.traj_opt)
            if ret_num != 0:
                logline('ret_num', ret_num)
                return end_action(1, us, config, plottings)
            control = utilsuite.jnumpify(lmppi.a_opt[0]) * config.norm_params[0, :2]/2
        

        ## Gym call
        control = [control[0] + np.random.normal(scale=config.added_noise[0]), 
                   control[1] + np.random.normal(scale=config.added_noise[1])]
        obs, step_reward, done, truncated, info = env.step(np.asarray([control]))        
        
        if config.render:
            ref_renderer.update(reference_traj[:, :2])
            if initial_loops:
                opt_traj_renderer.update(jit_device_get(mppi.traj_opt[:, :2]))
            else:
                opt_traj_renderer.update(jit_device_get(lmppi.traj_opt_c[:, :2]))
            # sampled_renderer.update(jit_concatenate(mppi.sampled_states))
        env.render()
        state_c_1 = obs['agent_0']['std_state'].copy()
        state_f_1 = state_c_1.copy()
        state_f_1[[0, 1, 4]] = obs['agent_0']['frenet_pose'].copy()
        # state_f_1[3] = state_c_1[3] * np.cos(state_c_1[6])
        
        
        laptime += config.sim_time_step
        if np.abs(state_f_1[1]) > config.half_width or lap_cnt == config.max_lap:
            if np.abs(state_f_1[1]) > config.half_width: 
                logline('out of bound', state_f_1)
                us.rec.ss_violation_record.append(lmppi.s_opt_ss_distance)
                us.rec.boundary_violation_record.append(lmppi.boundary_distance)
                us.rec.s_states = np.asarray(us.rec.s_states)
                us.rec.xy_states = np.asarray(us.rec.xy_states)
                us.rec.dyna_train_data.append(us.rec.xy_states)
                us.rec.a_opt_records.append(us.rec.a_opt_record)
                us.rec.mean_vels.append(np.mean(us.rec.s_states[:, 3]))
                us.rec.lamb_records.append(us.rec.lamb_record)
                us.rec.ss_average_record.append(np.mean(us.rec.ss_violation_record))
                us.rec.ss_max_record.append(np.max(us.rec.ss_violation_record))
                us.rec.boundary_average_record.append(np.mean(us.rec.boundary_violation_record))
                us.rec.boundary_max_record.append(np.max(us.rec.boundary_violation_record))
                us.rec.control_records.append(np.asarray(us.rec.control_record))
                if config.freeze_render_when_stop:
                    time.sleep(float("inf"))
                return end_action(2, us, config, plottings)
            logline('min laptime', np.min(us.rec.laptime_record))
            LOOP = False

        ## end of a lap
        if lap_cnt < info['lap_counts'][0] and laptime > 1:
            lap_cnt += 1
            # save_iteration(lap_cnt, value_fn, config, us)
            
            us.rec.ss_violation_record.append(lmppi.s_opt_ss_distance)
            us.rec.boundary_violation_record.append(lmppi.boundary_distance)
            us.rec.s_states = np.asarray(us.rec.s_states)
            us.rec.xy_states = np.asarray(us.rec.xy_states)
            us.rec.dyna_train_data.append(us.rec.xy_states)
            us.rec.a_opt_records.append(us.rec.a_opt_record)
            us.rec.laptime_record.append(laptime)
            us.rec.lamb_records.append(us.rec.lamb_record)
            us.rec.ss_average_record.append(np.mean(us.rec.ss_violation_record))
            us.rec.ss_max_record.append(np.max(us.rec.ss_violation_record))
            us.rec.boundary_average_record.append(np.mean(us.rec.boundary_violation_record))
            us.rec.boundary_max_record.append(np.max(us.rec.boundary_violation_record))
            us.rec.control_records.append(np.asarray(us.rec.control_record))
            us.rec.mean_vels.append(np.mean(us.rec.s_states[:, 3]))
            us.rec.mean_lambs.append(np.mean(np.array(us.rec.lamb_record), axis=0))
            if laptime < us.rec.min_laptime:
                us.rec.min_laptime = laptime
                us.rec.min_states = us.rec.xy_states
            
            pbar.update(1)
            pbar.set_description(f'laptime {laptime:.2f}, ss_vio: {np.mean(us.rec.ss_violation_record):.3f}, bd_vio: {np.max(us.rec.boundary_violation_record):.3f}')
            logline(lap_cnt, f'mean speed: {np.mean(us.rec.s_states[:, 3]):.2f}', f'laptime: {laptime:.2f}', 
                    f'lambs_opt: {np.mean(np.array(us.rec.lamb_record), axis=0)} {np.max(np.array(us.rec.lamb_record), axis=0)}', print_line=config.print_line)
            logline(f'average ss_violation: {np.mean(us.rec.ss_violation_record):.2f} max {np.max(us.rec.ss_violation_record):.2f}',
                f'average boundary_violation: {np.mean(us.rec.boundary_violation_record):.2f} max {np.max(us.rec.boundary_violation_record):.2f}', print_line=config.print_line)
            
            ## Train dynamic model
            if dyna_config.dyna_norm_params is not None and config.train_dyna_model:
                data_in, data_out, dyna_config = get_dyna_train_data(us.dp, us.rec.dyna_train_data, us.rec.control_records, dyna_config)
                train_data_len = 2000
                if data_in.shape[0] > train_data_len:
                    subsample_inds = utilsuite.get_subsample_inds(data_in.shape[0], train_data_len)
                    data_in = data_in[subsample_inds]
                    data_out = data_out[subsample_inds]
                infer_env.dyna_model.flax_train_state, dynamic_model_losses = infer_env.dyna_model.train(infer_env.dyna_model.flax_train_state,
                                                                                        data_out, data_in, max_epoch=300, loss_threshold=0.001)    
                logline('dynamic_model_losses', dynamic_model_losses[-1], print_line=config.print_line) 
            
            ## update safe set
            safe_set.add_lap(us.rec.laptimes, us.rec.s_states, us.rec.xy_states)
            if safe_set.get_new_safe_set():
                if value_fn.thread is not None:
                    value_fn.thread.join()
                    safe_set.update_ss_arr()
                value_fn.thread = threading.Thread(target=value_fn.train_valuefn, 
                                                    args=(safe_set.safe_set_frenet, 
                                                            config, value_fn.thread_ret))
                value_fn.thread.start()
                us.rec.safe_set.append(safe_set.safe_set_frenet[-1])
                us.rec.safe_set_xy.append(safe_set.safe_set_carti[-1])

            ## save safe set for quick load
            if len(us.rec.safe_set) == config.ss_size and (not config.quick_load): ## save initial safe set
                config.quick_load = 1
                us.rec.init_state = np.asarray(obs['agent_0']['state'])
                us.rec.save_onefile('safe_set', 'safe_set_xy', 'a_opt_records', 'control_records', 'dyna_train_data', 'init_state',
                                    save_dir=config.save_dir, filename='init')
                logline('init saved', print_line=config.print_line)
            
                
            if len(us.rec.dyna_train_data) >= config.ss_size + 5: 
                us.rec.control_records.pop(0)
                us.rec.dyna_train_data.pop(0)
            
#             # if lap_cnt > 0 and len(us.rec.safe_set) >= config.ss_size:
#             plottings.plot_speed_position_history(lap_cnt, laptime, True)
#             #     plottings.plot_trajectory_history(lap_cnt, laptime)
#             #     plottings.plot_laptime_speed(key_option, laptime)
#             #     plottings.plot_ss_violation(key_option, laptime)
#             #     plottings.plot_boundary_violation(key_option, laptime)
#             #     plottings.plot_speed_position(key_option, laptime)
#             #     plottings.plot_speed_position_history(lap_cnt, laptime)
#                 # plottings.plot_acceleration_position(key_option, laptime) 
#                 # plottings.plot_acceleration_position_history(lap_cnt, laptime)
#                 # plottings.plot_ss_lamb_history(lap_cnt, laptime)
#                 # plottings.plot_boundary_lamb_history(lap_cnt, laptime)
#                 # plottings.plot_lap_lambs(lap_cnt, laptime)
                
            
            laptime = 0.0
            us.rec.init('laptimes', # record per action laptime,
                        'control_record',
                        's_states',
                        'xy_states',
                        'a_opt_record',
                        'ss_violation_record',
                        'boundary_violation_record',
                        'lamb_record')
        else:
            us.rec.ss_violation_record.append(lmppi.s_opt_ss_distance)
            us.rec.boundary_violation_record.append(lmppi.boundary_distance)
            us.rec.laptimes.append(laptime)
            us.rec.s_states.append(state_f_1)
            us.rec.xy_states.append(state_c_0)
            if initial_loops:
                us.rec.a_opt_record.append(mppi.a_opt)
            else:
                us.rec.a_opt_record.append(lmppi.a_opt)
            us.rec.control_record.append(np.asarray(control))
            us.rec.lamb_record.append(np.asarray(lmppi.lambs_opt))  
            

    logline('Sim elapsed time:', np.sum(us.rec.laptime_record), 
          'Real elapsed time:', time.time() - start_time)
    if config.train_value_with_all_ss:
        return end_action(0, us, config, plottings, safe_set, value_fn, value_config)
    else:
        return end_action(0, us, config, plottings)

def end_action(ret_num, us, config, plottings, safe_set=None, value_fn=None, value_config=None):
    if ret_num == 1:
        return 1, us
    if len(us.rec.laptime_record) == 0:
        return 1, us
    else:
        plottings.plot_laptime_speed(0, np.min(us.rec.laptime_record), enable=True)
        plottings.plot_min_speed_position(0, us.rec.min_laptime, enable=True)
        plottings.plot_ss_violation(0, us.rec.min_laptime, enable=True)
        plottings.plot_boundary_violation(0, us.rec.min_laptime, enable=True)
    ## recording save
    us.rec.save_onefile('safe_set', 'safe_set_xy', 'a_opt_records', 
                'control_records', 'lamb_records', 'traj_opt',
                'laptime_record', 'ss_average_record', 'ss_max_record',
                'boundary_average_record', 'boundary_max_record',
                'min_laptime', 'min_states', 'time_record', save_dir=config.save_dir)
    if safe_set is not None:
        safe_set.safe_set_frenet = copy.deepcopy(us.rec.safe_set)
        safe_set.safe_set_carti = copy.deepcopy(us.rec.safe_set_xy)
        safe_set.update_ss_arr()        
        if value_fn.thread is not None:                                  
            value_fn.thread.join()
        value_fn.thread = threading.Thread(target=value_fn.train_valuefn, args=(safe_set.safe_set_frenet, config, 
                                                                                value_fn.thread_ret, value_config.max_epoch*2))
        value_fn.thread.start()
        value_fn.thread.join()
        value_fn.save_model(config.save_dir)
    return ret_num, us
    
def get_noise_truncated_normal(std, limit_mulitplier, size):
    return utilsuite.truncated_normal_sampler(mean=0, std=std, lower_bound=-std*limit_mulitplier, upper_bound=std*limit_mulitplier, size=size)

def save_iteration(lap_cnt, valuefn, config, us):
    save_dir = config.save_dir + f'iterations/iteration_{lap_cnt}/'
    us.mkdir(us, save_dir)
    valuefn.save_model(save_dir)
    config.save(save_dir + 'config.yaml')
    us.rec.save_onefile('safe_set', 'safe_set_xy', 'a_opt_records', 
                'control_records', 'lamb_records', 'traj_opt',
                'laptime_record', 'ss_average_record', 'ss_max_record',
                'boundary_average_record', 'boundary_max_record',
                'min_laptime', 'min_states', 'time_record', save_dir=save_dir)

@jax.jit
def jit_concatenate(sampled_states):
    return jax.device_get(jnp.concatenate(sampled_states[:, :, (0, 1, 4)]))[:, :2]

@jax.jit
def jit_device_get(array):
    return jax.device_get(array)

if __name__ == '__main__':
    main()
