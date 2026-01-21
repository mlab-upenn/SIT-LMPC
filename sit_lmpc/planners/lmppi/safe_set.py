import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from scipy.spatial import ConvexHull
from concurrent.futures import ThreadPoolExecutor
# from sklearn.neighbors import KernelDensity

def get_subsample_inds(length, subsample_num):
    if subsample_num is None:
        return np.arange(length)
    if subsample_num > length:
        subsample_num = length
    return np.random.permutation(length)[:subsample_num]

class SafeSet:
    def __init__(self, config, logline, track=None) -> None:
        self.logline = logline
        self.config = config
        self.track = track
        self.ss_hull_num_reduction = config.ss_hull_num_reduction
        self.value_dim = config.value_dim
        self.n_samples = config.n_samples
        self.n_steps = config.n_steps
        self.precompile_range = np.arange(self.ss_hull_num_reduction, config.ss_hull_precompile_len, self.ss_hull_num_reduction)
        
        self.lap_record_frenet = []
        self.lap_record_carti = []
        self.safe_set_frenet = []
        self.safe_set_carti = []
        self.ss_arr_max_len = self.config.ss_arr_max_len
        self.ss_select_max_len = self.config.ss_select_max_len
        
    def add_lap(self, time_lap, frenet_lap=None, carti_lap=None):
        values = np.asarray(time_lap)
        if frenet_lap is not None:
            self.lap_record_frenet.append(np.concatenate([np.asarray(frenet_lap), values[:, None]], axis=1))
            self.lap_record_carti.append(np.concatenate([np.asarray(carti_lap), values[:, None]], axis=1))
        else:
            self.lap_record_carti.append(np.concatenate([np.asarray(carti_lap), values[:, None]], axis=1))
            
    def get_new_safe_set(self):
        new_safe_set = False
        if len(self.lap_record_frenet) > 1:
            ss_extend_s = np.max(self.lap_record_frenet[-1][:, 3]) * self.config.ss_loop_extend_sec * \
                self.config.n_steps * self.config.sim_time_step # extend s based on max speed of last lap 
            extend_inds = np.where(self.lap_record_frenet[-1][:, 0] < ss_extend_s)
            extended_lap = self.lap_record_frenet[-1][extend_inds] 
            extended_lap[:, 0] += self.track.s_frame_max
            extended_lap = np.concatenate([self.lap_record_frenet[-2], extended_lap])                
            extended_lap[:len(self.lap_record_frenet[-2]), -1] -= np.max(extended_lap[:, -1])
            self.safe_set_frenet.append(extended_lap)
            extended_lap_xy = self.lap_record_carti[-1][extend_inds]
            extended_lap_xy = np.concatenate([self.lap_record_carti[-2], extended_lap_xy])
            extended_lap_xy[:, -1] = extended_lap[:, -1]
            self.safe_set_carti.append(extended_lap_xy)
            self.lap_record_frenet.pop(0)
            self.lap_record_carti.pop(0)
            new_safe_set = True
        elif self.config.state_predictor == 'point_mass':
            if len(self.lap_record_carti) >= 1:
                lap_xy = self.lap_record_carti[-1]   
                lap_xy[:len(self.lap_record_carti[-1]), -1] -= np.max(lap_xy[:, -1])
                self.safe_set_carti.append(lap_xy)
                new_safe_set = True
        while len(self.safe_set_carti) >= self.config.ss_size + 1: 
            self.safe_set_carti.pop(0)
        while len(self.safe_set_frenet) >= self.config.ss_size + 1: 
            self.safe_set_frenet.pop(0)
        return new_safe_set
    
    def update_ss_arr(self):
        if len(self.safe_set_frenet) > 0:
            self.ss_arr_frenet = np.concatenate(self.safe_set_frenet)
        if len(self.safe_set_carti) > 0:
            self.ss_arr_carti = np.concatenate(self.safe_set_carti)
        if self.ss_arr_frenet.shape[0] > self.ss_arr_max_len:
            subsample_inds = get_subsample_inds(self.ss_arr_frenet.shape[0], self.ss_arr_max_len)
            self.ss_arr_frenet = self.ss_arr_frenet[subsample_inds]
            self.ss_arr_carti = self.ss_arr_carti[subsample_inds]
        

    def update_convex_hull(self, ref_ss_states):
        xSS_hull_equations = ConvexHull(ref_ss_states[:, :self.value_dim]).equations
        if xSS_hull_equations.shape[0] % self.ss_hull_num_reduction != 0:
            xSS_hull_equations = xSS_hull_equations[:-(xSS_hull_equations.shape[0] % self.ss_hull_num_reduction)]
        self.xSS_hull_equations = jnp.array(xSS_hull_equations)
        self.len_xSS_hull = xSS_hull_equations.shape[0]
        # print(xSS_hull_equations.shape[0], ref_ss_states.shape)
        
    def compiled_states_r_sshull(self, states):
        r_sshull = self.compiled[self.len_xSS_hull](states, self.xSS_hull_equations)
        return r_sshull
    
    def compiled_trajs_r_sshull(self, states):
        r_sshull = self.compiled2[self.len_xSS_hull](states, self.xSS_hull_equations)
        return r_sshull
    
    def get_ss_dist(self, states):
        try:
            if self.n_samples == states.shape[0]:
                r_sshull = self.compiled[self.xSS_hull_equations.shape[0]](states[:, :self.value_dim], self.xSS_hull_equations)
            else:
                r_sshull = self.compiled2[self.xSS_hull_equations.shape[0]](states[:, :self.value_dim], self.xSS_hull_equations)
        except Exception as e:
            print(e)
            r_sshull = self.vmap_sshull_reward(states[:, :self.value_dim], self.xSS_hull_equations)
            print(self.xSS_hull_equations.shape[0])
        return r_sshull
    
    # def get_ss_density(self, ref_ss_s, states):
    #     # original ABC-LMPC implementation was with tophat kernel, but it doesn't work with the racecar
    #     self.kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(ref_ss_s[:, :self.value_dim])
    #     # self.kde = KernelDensity(kernel='tophat', bandwidth=self.config.ss_relaxation).fit(ref_ss_s[:, :self.value_dim])
    #     r_sshull = -self.kde.score_samples(states[:, -1, :self.value_dim])
    #     return r_sshull

    def find_ss_inrange(self, terminal_states, ss_arr, env_state):
        terminal_states = terminal_states[:, 0]
        terminal_states_max_states = np.max(terminal_states, axis=0)
        terminal_states_min_states = np.min(terminal_states, axis=0)
        speed = np.maximum(env_state[3], self.config.init_vel)
        terminal_states_max = terminal_states_max_states + speed * (self.config.ss_ref_step_interval+1) * self.config.sim_time_step
        terminal_states_min = terminal_states_min_states - speed * (self.config.ss_ref_step_interval-1) * self.config.sim_time_step
        ss_inds = np.where((ss_arr[:, 0] > terminal_states_min) & (ss_arr[:, 0] < terminal_states_max))
        if len(ss_inds[0]) > self.ss_select_max_len:
            # ss_inds = (ss_inds[0][:self.ss_select_max_len],)
            subsample_inds = get_subsample_inds(len(ss_inds[0]), self.ss_select_max_len)
            ss_inds = (ss_inds[0][subsample_inds],)
        return ss_inds
    
    def find_ss_inrange_zt(self, zt, ss_arr):
        # Find the closes num_points points in ss to zt
        speed = np.maximum(zt[3], self.config.init_vel)
        zt_s = zt[0] + speed * self.config.sim_time_step # Simply s + v * dt
        terminal_states_s = ss_arr[:, 0]
        min_norm = np.argpartition(np.abs(terminal_states_s - zt_s), self.ss_select_max_len)[:self.ss_select_max_len]
        ss_inds = (min_norm,)
        return ss_inds
    
    @partial(jax.jit, static_argnums=(0))
    def sshull_reward(self, state, xSS_hull_equations):
        """
        reward function for the state s with distance to the safe set
        """
        ss_distance = jnp.max(jnp.dot(xSS_hull_equations[:, :-1], state).T + xSS_hull_equations[:, -1], axis=-1)
        ss_distance = ss_distance * (ss_distance > 0)
        return ss_distance
    
    @partial(jax.jit, static_argnums=(0))
    def vmap_sshull_reward(self, state_frenet, xSS_hull_equations):
        return jax.vmap(self.sshull_reward, in_axes=(0, None))(
            state_frenet, xSS_hull_equations
        )
    
    def ss_hull_precompile(self, lambs_len):
        self.logline('Precompiling jit for ss hull')
        self.lambs_len = lambs_len
        dummy_input1 = jnp.ones((self.n_samples, self.value_dim))
        dummy_input2 = jnp.ones((lambs_len, self.value_dim))
        dummy_equations = [jnp.ones((ind, self.value_dim + 1)) for ind in self.precompile_range]
        compiled = {}
        compiled2 = {}
        with ThreadPoolExecutor() as pool:
            for x in dummy_equations:
                compiled[x.shape[0]] = jax.jit(self.vmap_sshull_reward, backend='gpu').lower(dummy_input1, x)
                compiled2[x.shape[0]] = jax.jit(self.vmap_sshull_reward, backend='gpu').lower(dummy_input2, x)
                compiled[x.shape[0]] = pool.submit(compiled[x.shape[0]].compile)
                compiled2[x.shape[0]] = pool.submit(compiled2[x.shape[0]].compile)
            self.compiled = {s: f.result() for s, f in compiled.items()}
            self.compiled2 = {s: f.result() for s, f in compiled2.items()}
        for ind in range(len(self.precompile_range)):
            arr1 = self.compiled[dummy_equations[ind].shape[0]](dummy_input1, dummy_equations[ind])
            arr2 = self.compiled2[dummy_equations[ind].shape[0]](dummy_input2, dummy_equations[ind])
            
    
            
            
    
    
    