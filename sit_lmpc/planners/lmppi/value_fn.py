import copy
import os
from functools import partial

import distrax
import jax
import jax.numpy as jnp
import numpy as np

from sit_lmpc.model_train import ModelTrain
from sit_lmpc.utils.trainer_jax import Trainer

# from models.networks import EnsembleBNN



class ValueFn:
    def __init__(self, us, value_config, config) -> None:
        self.us = us
        self.thread_ret = [None, None]
        self.thread = None
        if value_config.model_type == 'bnn':
            self.model = EnsembleBNN(value_config)
        else:
            self.model = ModelTrain(value_config)
        if value_config.model_type == 'nf':
            self.dist_narrow = distrax.MultivariateNormalDiag(jnp.zeros(1), jnp.ones(1)/10)
        self.value_config = copy.deepcopy(value_config)
        self.config = copy.deepcopy(config)
        self.value_dim = config.value_dim
        self.state_dim = config.state_dim
        self.trainer = Trainer('valuefn', self.config.save_dir)

    def load_valuefn(self, safe_set, config):
        self.config = copy.deepcopy(config)
        safe_set = copy.deepcopy(safe_set)
        safe_set_cat = np.concatenate(safe_set)
        self.data_range = np.array([np.min(safe_set_cat, axis=0), np.max(safe_set_cat, axis=0)])
        self.model.flax_train_state, _ = self.trainer.load_state(self.model.flax_train_state, 
                                                              path=os.path.abspath(self.config.save_dir)+"/valuefn_model_best",
                                                              abs_path=True)
        return self.model, self.config
    
    @partial(jax.jit, static_argnums=(0))
    def get_value(self, flax_train_state, sampled_states, rng_key, data_range):
        ## Get the value model prediction

        data_in = sampled_states[:, :self.config.value_dim]
        if self.value_config.model_type == 'nf':
            data_in = (data_in - data_range[0, :self.value_dim]) / (data_range[1, :self.value_dim] - data_range[0, :self.value_dim])
        # z = self.dist_narrow.sample(seed=rng_key, 
        #                             sample_shape=(sampled_states.shape[0] * self.value_config.n_sample)).reshape(sampled_states.shape[0], self.value_config.n_sample)
        # data_in = jnp.concatenate([data_in, z], axis=1)
        sampled_value = self.model.test(flax_train_state, 
                            data_in,
                            rng_key)
        if self.value_config.model_type == 'nf':
            sampled_value, sampled_value_var = sampled_value
            # sampled_value = (sampled_value + 1) / 2
            sampled_value = sampled_value * (data_range[1, -1] - data_range[0, -1]) + data_range[0, -1]
        sampled_value = jnp.clip(sampled_value, data_range[0, -1], data_range[1, -1])
        # if self.value_config.model_type != 'nf' or sampled_value_var is None:
        #     sampled_value_var = jnp.zeros((self.config.n_samples, self.config.n_steps))

        return sampled_value
    
    def save_model(self, path=None):
        if path is None:
            path = self.config.save_dir
        self.trainer.save_state(self.model.flax_train_state, path=os.path.abspath(path)+"/")

    def train_valuefn(self, safe_set, config, value_learning_lmppi_ret, max_epoch=0):
        self.config = copy.deepcopy(config)
        safe_set = copy.deepcopy(safe_set)
        max_epoch = self.value_config.max_epoch if max_epoch == 0 else max_epoch
        
        safe_set_cat = np.concatenate(safe_set)
        self.data_range = np.array([np.min(safe_set_cat, axis=0), np.max(safe_set_cat, axis=0)])
        if self.value_config.model_type == 'bnn':
            self.data_range[:, -1] -= self.data_range[0, -1]
            
        if self.value_config.model_type == 'nf':
            data_out = safe_set_cat[:, -1:]
            data_in = safe_set_cat[:, :-1]
            if self.config.state_predictor == 'point_mass':
                data_out, data_in = self.add_boundary_pointmass(safe_set_cat, 20)
            else:
                data_out, data_in = self.add_boundary_racecar(safe_set_cat, 100)
            data_out = (data_out - self.data_range[0, -1]) / (self.data_range[1, -1] - self.data_range[0, -1])
            # data_out = data_out * 2 - 1
            data_in = (data_in - self.data_range[0, :self.value_dim]) / (self.data_range[1, :self.value_dim] - self.data_range[0, :self.value_dim])
        elif self.value_config.model_type == 'bnn':
            data_out, data_in = self.compile_bnn_data(safe_set)  
        
        self.model.flax_train_state, losses = self.model.train(self.model.flax_train_state, 
                                                                data_out,
                                                                data_in,
                                                                max_epoch=max_epoch, loss_threshold=0.0001)
        self.trainer.save_state(self.model.flax_train_state, path=os.path.abspath(self.config.save_dir)+"/")
        
        self.us.logline('losses', losses[-1], print_line=self.config.print_line)
        # value_learning_lmppi_ret[0] = self.model
        # value_learning_lmppi_ret[1] = self.config
        return self.model, self.config
    
    def compile_bnn_data(self, safe_set):
        data_outs = []
        data_ins = []
        for ind_ss in range(len(safe_set)):
            ss_arr = safe_set[ind_ss]
            if self.config.state_predictor == 'point_mass':
                data_out, data_in = self.add_boundary_pointmass(ss_arr, 5)
            else:
                data_out, data_in = self.add_boundary_racecar(ss_arr, 20)
            
            if ind_ss == 0:
                fix_length = data_in.shape[0]
            else:
                if data_in.shape[0] < fix_length:
                    original_len = data_in.shape[0]
                    data_in = jnp.concatenate([data_in, jnp.zeros((fix_length-original_len, self.value_dim))], axis=0)
                    data_out = jnp.concatenate([data_out, jnp.zeros((fix_length-original_len, 1))], axis=0)
                elif data_in.shape[0] > fix_length:
                    data_in = data_in[:fix_length]
                    data_out = data_out[:fix_length]
            
            data_outs.append(data_out)
            data_ins.append(data_in)
        data_outs = jnp.array(data_outs)
        data_ins = jnp.array(data_ins)        
        return data_outs, data_ins
    
    def add_boundary_racecar(self, ss_arr, n_boundary_data=20):
        boundary_data = np.zeros((n_boundary_data, self.value_dim + 1)) # create train data for out of boundary values
        boundary_data[:, 0] = np.random.uniform(self.data_range[0, 0], self.data_range[0, 1], size=boundary_data.shape[0])
        boundary_data[:, 1] = np.random.uniform(self.data_range[1, 0], self.data_range[1, 1], size=boundary_data.shape[0])
        boundary_data[:, 1][np.where(boundary_data[:, 1] <= 0)] -= self.config.half_width
        boundary_data[:, 1][np.where(boundary_data[:, 1] >= 0)] += self.config.half_width
        for ind in range(2, self.value_dim):
            boundary_data[:, ind] = np.random.uniform(self.data_range[0, ind], self.data_range[1, ind], size=boundary_data.shape[0])
        boundary_data[:, self.value_dim] = np.min(ss_arr[:, self.value_dim])
        boundary_data = jnp.asarray(boundary_data)
        if self.value_config.model_type == 'bnn':
            ss_arr[:, -1] += self.data_range[1, -1]
        data_out = jnp.concatenate([ss_arr[:, -1:], boundary_data[:, -1:]], axis=0)
        data_in = jnp.concatenate([ss_arr[:, :self.value_dim], boundary_data[:, :self.value_dim]], axis=0)
        return data_out, data_in
    
    def add_boundary_pointmass(self, ss_arr, n_boundary_data=5):
        boundary_data = np.zeros((n_boundary_data, self.value_dim + 1)) # create train data for out of boundary values
        boundary_data[:, 0] = np.random.uniform(-2, 0, size=boundary_data.shape[0])
        boundary_data[:, 1] = np.random.uniform(-2, 2, size=boundary_data.shape[0])
        
        boundary_data2 = np.zeros((n_boundary_data, self.value_dim + 1)) # create train data for out of boundary values
        boundary_data2[:, 0] = np.random.uniform(-2, 2, size=boundary_data.shape[0])
        boundary_data2[:, 1] = np.random.uniform(-2, 0, size=boundary_data.shape[0])
        
        goal_data = np.zeros((n_boundary_data, self.value_dim + 1)) # create train data for out of boundary values
        goal_data[:, 0] = self.config.target_pos[0]
        goal_data[:, 1] = self.config.target_pos[1]

        boundary_data = np.concatenate([boundary_data, boundary_data2, goal_data], axis=0)
        for ind in range(2, self.value_dim):
            boundary_data[:, ind] = np.random.uniform(self.data_range[0, ind], self.data_range[1, ind], size=boundary_data.shape[0])
        # if self.value_config.model_type == 'bnn':
        ss_arr[:, -1] += self.data_range[1, -1]
        boundary_data[:, self.value_dim] = np.min(ss_arr[:, self.value_dim])
        boundary_data[-10:, self.value_dim] = np.max(ss_arr[:, self.value_dim])
        # boundary_data[:, self.value_dim] = self.data_range[0, -1]
        # boundary_data[-10:, self.value_dim] = self.data_range[1, -1]
        
        data_out = jnp.concatenate([ss_arr[:, -1:], boundary_data[:, -1:]], axis=0)
        data_in = jnp.concatenate([ss_arr[:, :self.value_dim], boundary_data[:, :self.value_dim]], axis=0)
        return data_out, data_in