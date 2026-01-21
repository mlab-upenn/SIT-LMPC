import os, sys
import jax
import jax.numpy as jnp
import numpy as np
import copy
from functools import partial

import utilsuite
from utilsuite import jax_jit
from sit_lmpc.planners.dynamics_models.dynamics_models_jax import *
from sit_lmpc.model_train import ModelTrain
from sit_lmpc.utils.trainer_jax import Trainer

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

class NNInfer:
    def __init__(self, dyna_config, params) -> None:
        self.dyna_config = copy.deepcopy(dyna_config)
        self.jrng = utilsuite.oneLineJaxRNG(dyna_config.random_seed)
        self.dyna_model = ModelTrain(dyna_config)
        self.trainer = Trainer(dyna_config.exp_name, dyna_config.savedir)
        self.dp = utilsuite.DataProcessor()
        self.params = params
        self.dyna_model.flax_train_state, dynamic_model_losses = self.trainer.load_state(self.dyna_model.flax_train_state, [], 'best', dyna_config.savedir)
        
    def update_fn_dynamics_ST(self, state, control, dyna_model, rng_key, dyna_norm_params):
        # if dyna_norm_params is not None:
        #     self.dyna_config.dyna_norm_params = dyna_norm_params
        
        data_in = jnp.array([self.dp.runtime_normalize(state[2], dyna_norm_params[0]),
                             self.dp.runtime_normalize(state[3], dyna_norm_params[1]),
                             self.dp.runtime_normalize(state[5], dyna_norm_params[2]),
                             self.dp.runtime_normalize(state[6], dyna_norm_params[3]),
                             self.dp.runtime_normalize(control[0], dyna_norm_params[8]),
                             self.dp.runtime_normalize(control[1], dyna_norm_params[9])
                            ]).T
        
        if self.dyna_config.model_type == 'nf':
            data_in = data_in[None, :]
            dyna_ret, dyna_ret_var = dyna_model.test(dyna_model.flax_train_state, data_in, rng_key)
            dyna_ret = dyna_ret[0]
        elif self.dyna_config.model_type == 'nn':
            dyna_ret = dyna_model.test(dyna_model.flax_train_state, data_in, rng_key)
        dyna_ret = jnp.clip(dyna_ret, 0, 1)

        dyna_ret = jnp.array([self.dp.de_normalize(dyna_ret[0], dyna_norm_params[4]),
                              self.dp.de_normalize(dyna_ret[1], dyna_norm_params[5]),
                              self.dp.de_normalize(dyna_ret[2], dyna_norm_params[6]),
                              self.dp.de_normalize(dyna_ret[3], dyna_norm_params[7])
                            ]).T

        return (state + self.vehicle_dynamics_st(state, dyna_ret) * self.dyna_config.DT, 0, dyna_ret_var)  


    @partial(jax.jit, static_argnums=(0,))
    def vehicle_dynamics_st(self, x, predict):
        """
        Single Track Dynamic Vehicle Dynamics.

            Args:
                x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5, x6, x7)
                    x1: x position in global coordinates
                    x2: y position in global coordinates
                    x3: steering angle of front wheels
                    x4: velocity in x direction
                    x5: yaw angle
                    x6: yaw rate
                    x7: slip angle at vehicle center
                u (numpy.ndarray (2, )): control input vector (u1, u2)
                    u1: steering angle velocity of front wheels
                    u2: longitudinal acceleration

            Returns:
                f (numpy.ndarray): right hand side of differential equations
        """
    
        s_min = self.params[8]  # minimum steering angle [rad]
        s_max = self.params[9]  # maximum steering angle [rad]
        sv_min = self.params[10]  # minimum steering velocity [rad/s]
        sv_max = self.params[11]  # maximum steering velocity [rad/s]
        v_switch = self.params[12]  # switching velocity [m/s]
        a_max = self.params[13]  # maximum absolute acceleration [m/s^2]
        v_min = self.params[14]  # minimum velocity [m/s]
        v_max = self.params[15]  # maximum velocity [m/s]
        
        # constraints
        u = jnp.array([steering_constraint(x[2], predict[0], s_min, s_max, sv_min, sv_max), 
                       accl_constraints(x[3], predict[1], v_switch, a_max, v_min, v_max)])
        # u = predict[0:2]

        # system dynamics
        f = jnp.array([x[3]*jnp.cos(x[6] + x[4]),
            x[3]*jnp.sin(x[6] + x[4]),
            u[0],
            u[1],
            x[5],
            predict[2],
            predict[3]])

        return f

class InferEnv():
    def __init__(self, track, config, DT, params_dict,
                 jrng=None, dyna_config=None) -> None:
        self.a_shape = 2
        self.track = track
        self.DT = DT
        self.config = config
        self.jrng = utilsuite.oneLineJaxRNG(config.random_seed) if jrng is None else jrng
        self.norm_params = config.norm_params
        self.dyna_config = dyna_config

        self.Ddt = self.config.integrator_timestep if hasattr(self.config, 'integrator_timestep') else 0.01
        self.loop_times = int(self.DT / self.Ddt)
        # self.mb_dyna_pre = None
        print('InferEnv Model:', self.config.state_predictor)
        
        self.sequential_params_st = ['mu', 'C_Sf', 'C_Sr', 'lf', 'lr', 'h', 'm', 'I',
                                    's_min', 's_max', 'sv_min', 'sv_max', 'v_switch', 
                                    'a_max', 'v_min', 'v_max', 'width', 'length']
        self.params = self.update_params(params_dict)
        
        def RK4_fn(x0, u, Ddt, vehicle_dynamics_fn, args):
            # return x0 + vehicle_dynamics_fn(x0, u, *args) * Ddt # Euler integration
            # RK4 integration
            k1 = vehicle_dynamics_fn(x0, u, *args)
            k2 = vehicle_dynamics_fn(x0 + k1 * 0.5 * Ddt, u, *args)
            k3 = vehicle_dynamics_fn(x0 + k2 * 0.5 * Ddt, u, *args)
            k4 = vehicle_dynamics_fn(x0 + k3 * Ddt, u, *args)
            return x0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * Ddt
        
        if self.config.state_predictor == 'point_mass':
            def update_fn(x, u, params, loop_times=self.loop_times, integrator_timestep=self.Ddt):
                x_k = x[:4]
                def step_fn(i, carry):
                    x0, params = carry
                    return (RK4_fn(x0, u, integrator_timestep, point_mass_dynamics_jax, (params,)), params)
                x_k, _ = jax.lax.fori_loop(0, loop_times, step_fn, (x_k, params))
                x1 = x.at[:4].set(x_k)
                return (x1, 0, x1-x)
            self.update_fn = jax_jit(update_fn, static_argnums=(3, 4))
            
        elif self.config.state_predictor == 'dynamic_ST':
            def update_fn(x, u, params, loop_times=self.loop_times, integrator_timestep=self.Ddt):
                def step_fn(i, x0):
                    return RK4_fn(x0, u, integrator_timestep, vehicle_dynamics_st, (params,))
                x1 = jax.lax.fori_loop(0, loop_times, step_fn, x)
                return (x1, 0, x1-x)
            self.update_fn = jax_jit(update_fn, static_argnums=(3, 4))
            
        elif self.config.state_predictor == 'kinematic_ST':
            def update_fn(x, u, params, loop_times=self.loop_times, integrator_timestep=self.Ddt):
                x_k = x[:5]
                def step_fn(i, x0):
                    return RK4_fn(x0, u, integrator_timestep, vehicle_dynamics_ks, (params,))
                x_k, _ = jax.lax.fori_loop(0, loop_times, step_fn, (x_k, params))
                x1 = x.at[:5].set(x_k)
                return (x1, 0, x1-x)
            self.update_fn = jax_jit(update_fn, static_argnums=(3, 4))
            
        elif self.config.state_predictor == 'nn_dynamic_ST':
            self.nn_infer = NNInfer(dyna_config, self.params)
            self.dyna_model = self.nn_infer.dyna_model
            self.dyna_config = self.nn_infer.dyna_config
            def update_fn(x, u, rng_key, dyna_norm_params):
                return self.nn_infer.update_fn_dynamics_ST(x, u, self.dyna_model, rng_key, dyna_norm_params)
            self.update_fn = update_fn

        # if self.config.state_predictor == 'mb':
        #     def update_fn(x, u, rng_key):
        #         x1 = x.copy()
        #         def step_fn(i, x0):
        #             return x0 + vehicle_dynamics_mb(x0, u) * 0.001
        #         x1 = jax.lax.fori_loop(0, int(self.DT/0.001), step_fn, x1)
        #         return (x1, 0, x1-x)
        #     self.update_fn = update_fn
    
    def update_params(self, params_dict):
        """
        Update the parameters of the environment.
        """
        if self.config.state_predictor in ['dynamic_ST', 'kinematic_ST', 'nn_dynamic_ST']:
            return jnp.array([params_dict[param] for param in self.sequential_params_st])
            

    @jax_jit(static_argnums=(0,))
    def step(self, x, u, params, rng_key=None, dyna_norm_params=None):
        if self.config.state_predictor == 'nn_dynamic_ST':
            return self.update_fn(x, u, rng_key, dyna_norm_params)
        else:
            return self.update_fn(x, u * self.norm_params[0, :2]/2, params)
    
    @partial(jax.jit, static_argnums=(0,))
    def reward_fn_sey(self, s, reference):
        """
        reward function for the state s with respect to the reference trajectory
        """
        # gamma = 0.8
        # gamma_vec = jnp.array([gamma ** i for i in range(reference.shape[0] - 1)])
        sey_cost = -jnp.linalg.norm(reference[1:, 4:6] - s[:, :2], ord=1, axis=1)
        # vel_cost = -jnp.linalg.norm(reference[1:, 3] - s[:, 3])
        # yaw_cost = -jnp.abs(jnp.sin(reference[1:, 4]) - jnp.sin(s[:, 4])) - \
        #     jnp.abs(jnp.cos(reference[1:, 4]) - jnp.cos(s[:, 4]))
            
        return sey_cost
    
    @partial(jax.jit, static_argnums=(0,))
    def reward_fn_xy(self, state, reference):
        """
        reward function for the state s with respect to the reference trajectory
        """
        # gamma = 0.8
        # gamma_vec = jnp.array([gamma ** i for i in range(reference.shape[0] - 1)])
        xy_cost = -jnp.linalg.norm(reference[1:, :2] - state[:, :2], ord=1, axis=1)
        vel_cost = -jnp.linalg.norm(reference[1:, 2] - state[:, 3])
        yaw_cost = -jnp.abs(jnp.sin(reference[1:, 3]) - jnp.sin(state[:, 4])) - \
            jnp.abs(jnp.cos(reference[1:, 4]) - jnp.cos(state[:, 4]))
            
        # return 20*xy_cost + 15*vel_cost + 1*yaw_cost
        return xy_cost
    
    # def state_st2infer(self, st_state):
    #     return jnp.array([st_state[2], 
    #                     st_state[3] * jnp.cos(st_state[6]),
    #                     st_state[5],
    #                     st_state[3] * jnp.sin(st_state[6])])
        
    
    # def state_st2nf(self, st_state):
    #     return np.array([st_state[0], st_state[1], st_state[2],
    #                     st_state[3] * np.cos(st_state[6]),
    #                     st_state[4], st_state[5],
    #                     st_state[3] * np.sin(st_state[6])])
        
    
    # def state_nf2st(self, nf_state):
    #     return np.array([nf_state[0], nf_state[1], nf_state[2],
    #                     np.sqrt(nf_state[3] ** 2 + nf_state[6] ** 2),
    #                     nf_state[4], nf_state[5],
    #                     np.arctan2(nf_state[6], nf_state[3])])
        
    # def state_mb2st(self, mb_state):
    #     return np.array([mb_state[0], mb_state[1], mb_state[2],
    #                     np.sqrt(mb_state[3] ** 2 + mb_state[10] ** 2),
    #                     mb_state[4], mb_state[5],
    #                     np.arctan2(mb_state[10], mb_state[3])])
        
        
    # def state_mb2nf(self, mb_state):
    #     return np.array([mb_state[0], mb_state[1], mb_state[2],
    #                     mb_state[3], mb_state[4], mb_state[5],
    #                     mb_state[10]])
        
    
    # def state_nf2mb(self, mb_state, nf_state):
    #     mb_state[0:6] = nf_state[0:6]
    #     mb_state[10] = nf_state[6]
    #     return mb_state
    
    
    # def state_nf2infer(self, mb_state):
    #     return jnp.array([mb_state[2], mb_state[3], mb_state[5], mb_state[6]])
        

    

    
