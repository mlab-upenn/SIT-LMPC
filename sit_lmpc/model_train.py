import os
import jax
import optax
import distrax
import numpy as np
from functools import partial
from flax import linen as nn
import jax.numpy as jnp
import flax.training.train_state as flax_TrainState
import tqdm
import utilsuite

from sit_lmpc.models.nsf import NeuralSplineFlow
from sit_lmpc.models.networks import MLP, BayesianPNN
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
class ModelTrain:
    def __init__(self, config) -> None:
        self.config = config
        self.pe = utilsuite.PositionalEncoding_jax(self.config.pe_level)
        self.dp = utilsuite.DataProcessor()
        self.jrng = utilsuite.oneLineJaxRNG(config.random_seed)
        self.timer = utilsuite.Timer()
        self.iterator = range if hasattr(self.config, 'silent') and self.config.silent else tqdm.trange

        self.x_init = jnp.zeros((config.batchsize, config.n_dim))
        self.x_context = jnp.zeros((config.batchsize, config.n_context))
        
        if config.model_type == 'nf':
            
            self.dist_narrow = distrax.MultivariateNormalDiag(jnp.zeros(1), jnp.ones(1)/10)
            self.dist = distrax.MultivariateNormalDiag(jnp.zeros(1), jnp.ones(1))
            config.activation = 'gelu' if config.activation is None else config.activation
            config.n_bins = 8 if config.n_bins is None else config.n_bins
            
            self.model = NeuralSplineFlow(n_dim=config.n_dim, n_context=config.n_context, 
                            hidden_dims=[config.hidden_dims, config.hidden_dims], 
                            n_transforms=config.layer_num, activation=config.activation, n_bins=config.n_bins)
            params = self.model.init(self.jrng.new_key(), self.x_init, self.x_context)    
            
        elif config.model_type == 'nn':
            self.model = MLP(out_dims=config.n_dim, hidden_dims=config.hidden_dims, layer_num=config.layer_num)
            params = self.model.init(self.jrng.new_key(), self.x_context)
        elif config.model_type == 'bnn':
            self.model = BayesianPNN(input_dim=config.n_context, output_features=config.n_dim, 
                                     hidden_features=config.hidden_dims, layer_num=config.layer_num)
            params = self.model.init(self.jrng.new_key(), self.x_context, self.jrng.new_key())
        print(config.model_type, 'num of params:', sum(x.size for x in jax.tree.leaves(params)))
        # self.z = self.dist_narrow.sample(seed=self.jrng.new_key(), sample_shape=(1 * self.config.n_sample))
           
        # schedule = optax.cosine_decay_schedule(
        #             config.lr, 500, alpha=0.01
        #             )
        # def schedule(count):
        #     ret = jax.lax.select(count > 2000, 1e-5, 1e-4)
        #     return ret
        
        self.flax_train_state = flax_TrainState.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optax.chain(optax.clip_by_global_norm(8), 
                        optax.adam(learning_rate=config.lr)),
        )
        
        
        
        # model_merit_fn = lambda info: info[1]
        # self.trainer = Trainer(EXP_NAME, config.savedir,
        #             max_epoch=config.max_epoch, best_fn=model_merit_fn, 
        #             rl_schedule_epoch=[100, 300], rl_schedule_gamma=0.1,
        #             info_template=np.zeros(8), initial_lr=config.lr)
        # self.trainer.is_done()
        self.epoch_info = np.zeros(8)
        # self.flax_train_state, self.epoch_info = self.trainer.load_state(self.flax_train_state, 
        #                                                             self.epoch_info, 
        #                                                             save_name='last')
        
    
    @partial(jax.jit, static_argnums=(0,))
    def train_step_nf(self, state, y_gt, context):
        def loss_fn(params):
            log_prob = state.apply_fn(
                params, y_gt, context
            )
            loss = -log_prob.mean()
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        
        state = state.apply_gradients(grads=grads)
        return state, loss
    
    def train(self, state, data_out, data_in, max_epoch=300, loss_threshold=0.0001):
        
        # params = self.model.init(self.jrng.new_key(), self.x_init, self.x_context)
        # state = flax_TrainState.TrainState.create(
        #     apply_fn=self.model.apply,
        #     params=params,
        #     tx=optax.chain(optax.clip_by_global_norm(8), 
        #                 optax.adam(learning_rate=self.config.lr)),
        # )

        losses = []
        for ind in self.iterator(max_epoch):         
            batchsize = np.minimum(self.config.batchsize, data_in.shape[0])
            train_perms = utilsuite.generate_perms(None, data_in.shape[0], batchsize)
            losses_batch = []
            for perms in train_perms:
                y_gt = jnp.asarray(data_out[perms])
                context = jnp.asarray(data_in[perms])
                if self.config.pe_level > 0:
                    context = self.pe.batch_encode(context)
                if self.config.model_type == 'nf':
                    state, loss = self.train_step_nf(state, y_gt, context)
                elif self.config.model_type == 'nn':
                    state, loss = self.model.train_step_nn(state, y_gt, context)
                elif self.config.model_type == 'bnn':
                    state, loss = self.model.train_step_bnn(state, y_gt, context, self.jrng.new_key())
                losses_batch.append(loss)
            mean_loss = jnp.mean(jnp.asarray(losses_batch)).astype(float)
            losses.append(mean_loss)
            # print(f"{ind}/{max_epoch}", mean_loss)
            # if mean_loss < loss_threshold:
            #     break
        # self.trainer.step(state, self.epoch_info)
        return state, losses

    @partial(jax.jit, static_argnums=(0,))
    def test(self, state, data_in, rng_key=None):
        
        def test_step_nn(state, context):
            preds = state.apply_fn(state.params, context)
            return preds
        
        def test_step_bnn(state, context, rng):
            preds, _ = state.apply_fn(state.params, context, rng)
            return preds
        
        def test_step_nf(rng_key, context):
            # if rng_key is None:
            #     z = context[:, self.config.n_context:].reshape(-1, self.config.n_dim)
            #     context = context[:, :self.config.n_context]
            # else:
            z = self.dist_narrow.sample(seed=rng_key, sample_shape=(context.shape[0] * self.config.n_sample))
            # z = jnp.zeros((context.shape[0] * self.config.n_sample, 1))
            context_batch = context[None, :, :].repeat(self.config.n_sample, 0)
            context_batch = context_batch.reshape(-1, context.shape[-1])
            samples = self.model.apply(state.params, z, context_batch, method=self.model.sample)
            # samples = self.pe.batch_decode2(samples)
            
            samples = samples.reshape(self.config.n_sample, -1, samples.shape[-1])
            samples_mean = samples.mean(axis=0)
            # z2 = self.dist2.sample(seed=rng_key, sample_shape=(context_batch.shape[0]))
            # samples = self.model.apply(state.params, z2, context_batch, method=self.model.sample)
            # samples = samples.reshape(self.config.n_sample, -1, samples.shape[-1])
            return samples_mean, jnp.var(samples, axis=0)
            # return samples_mean, None
        
        context = data_in
        if self.config.pe_level > 0:
            context = self.pe.batch_encode(context)
        if self.config.model_type == 'nf':
            ret = test_step_nf(rng_key, context)
        elif self.config.model_type == 'nn':
            ret = test_step_nn(state, context)
        elif self.config.model_type == 'bnn':
            ret = test_step_bnn(state, context, rng_key)
                
        return ret

    
    