import jax
import jax.numpy as jnp
import numpy as np

import utilsuite
from sit_lmpc.utils.trainer_jax import Trainer
from sit_lmpc.model_train import ModelTrain

class dynaConfig(utilsuite.ConfigYAML):
    EXP_NAME = 'nfST'
    exp_name = EXP_NAME
    # datadir = '/workspace/data/lmppi/dyna_rand_uniform/'
    datadir = '/media/lucerna/DATA/dyna_rand_uniform/'
    savedir = datadir + EXP_NAME + '/'
    latent_size = 1
    hidden_dims = 128
    layer_num = 3
    n_bins = 6
    pe_level = 0
    batchsize = 100
    test_batchsize = 100000
    lr = 1e-4
    n_dim = 4
    n_context = 6
    n_sample = 10
    max_epoch = 300
    model_type = 'nf'
    loop_num = 0
    DT = 0.1
    activation = 'swish'
    sigmoid_slope_last = 1
    
def get_dyna_train_data(dp, dyna_train_data, control_records, dyna_config):
    dyna_train_xu = []
    dyna_train_dx = []
    for ind in range(len(dyna_train_data)):
        data_length = len(control_records[ind][:-1])
        dyna_train_xu.append(np.concatenate([dyna_train_data[ind][:data_length, (2, 3, 5, 6)],
                                            control_records[ind][:data_length]], axis=1))
        dyna_train_dx.append((dyna_train_data[ind][1:data_length+1, (2, 3, 5, 6)] - dyna_train_data[ind][:data_length, (2, 3, 5, 6)])/dyna_config.DT)
    dyna_train_xu = np.concatenate(dyna_train_xu)
    dyna_train_dx = np.concatenate(dyna_train_dx)
    dyna_norm_params = np.zeros(dyna_config.dyna_norm_params.shape)
    for ind in range(4):
        dyna_norm_params[ind] = np.array([np.max(dyna_train_xu[:, ind]) - np.min(dyna_train_xu[:, ind]), 
                                                    np.min(dyna_train_xu[:, ind])])
    for ind in range(4):
        dyna_norm_params[ind+4] = np.array([np.max(dyna_train_dx[:, ind]) - np.min(dyna_train_dx[:, ind]), 
                                                    np.min(dyna_train_dx[:, ind])])
    for ind in range(2):
        dyna_norm_params[ind+8] = np.array([np.max(dyna_train_xu[:, ind+4]) - np.min(dyna_train_xu[:, ind+4]), 
                                                    np.min(dyna_train_xu[:, ind+4])])

    dyna_norm_params = dp.find_larger_normal_params(dyna_norm_params, dyna_config.dyna_norm_params)
    dyna_config.dyna_norm_params = dyna_norm_params.copy()
    
    
    data_in = jnp.concatenate([dp.runtime_normalize(dyna_train_xu[:, 0:1], dyna_norm_params[0]),
                            dp.runtime_normalize(dyna_train_xu[:, 1:2], dyna_norm_params[1]),
                            dp.runtime_normalize(dyna_train_xu[:, 2:3], dyna_norm_params[2]),
                            dp.runtime_normalize(dyna_train_xu[:, 3:4], dyna_norm_params[3]),
                            dp.runtime_normalize(dyna_train_xu[:, 4:5], dyna_norm_params[8]),
                            dp.runtime_normalize(dyna_train_xu[:, 5:6], dyna_norm_params[9])
                        ], axis=1)
    data_out = jnp.concatenate([dp.runtime_normalize(dyna_train_dx[:, 0:1], dyna_norm_params[4]),
                              dp.runtime_normalize(dyna_train_dx[:, 1:2], dyna_norm_params[5]),
                              dp.runtime_normalize(dyna_train_dx[:, 2:3], dyna_norm_params[6]),
                              dp.runtime_normalize(dyna_train_dx[:, 3:4], dyna_norm_params[7])
                            ], axis=1)

    return data_in, data_out, dyna_config
    
def main():
    ct = utilsuite.coloredText()
    dyna_config = dynaConfig()
    dyna_config.load(dyna_config.datadir + 'config.yaml')
    print('datadir', dyna_config.datadir)
    dyna_config.random_seed = 0
    dynamic_model = ModelTrain(dyna_config)
    us, _ = utilsuite.utilitySuite()
    trainer = Trainer(dyna_config.exp_name, dyna_config.savedir)
    jrng = utilsuite.oneLineJaxRNG(dyna_config.random_seed)
    
    train_data = np.load(dyna_config.datadir + 'train_data.npz', allow_pickle=True)
    train_states = train_data['train_states']
    train_controls = train_data['train_controls']
    train_dynamics = train_data['train_dynamics']
    
    train_states_norm = train_states.copy()[0]
    train_dynamics_norm = train_dynamics.copy()[0]
    train_controls_norm = train_controls.copy()[0]

    for ind in range(4):
        train_states_norm[:, 0, ind] = us.dp.runtime_normalize(train_states[0, :, 0, ind], dyna_config.dyna_norm_params[ind])
    for ind in range(4):    
        train_dynamics_norm[:, 0, ind] = us.dp.runtime_normalize(train_dynamics[0, :, 0, ind], dyna_config.dyna_norm_params[ind+4])
    for ind in range(2):
        train_controls_norm[:, 0, ind] = us.dp.runtime_normalize(train_controls[0, :, 0, ind], dyna_config.dyna_norm_params[ind+8])
    print('dyna_config.dyna_norm_params', dyna_config.dyna_norm_params)
    
    dyna_train_dx = jnp.array(train_dynamics_norm[:, 0, :])
    dyna_train_xu = jnp.array(np.concatenate([train_states_norm[:, 0, :], train_controls_norm[:, 0, :]], axis=1))
    print(dyna_train_dx.shape, dyna_train_xu.shape)
    us.timer.tic()
    dynamic_model.flax_train_state, dynamic_model_losses = dynamic_model.train(dynamic_model.flax_train_state,
                                                                                        dyna_train_dx,
                                                                                        dyna_train_xu, 
                                                                                        max_epoch=dyna_config.max_epoch, 
                                                                                        loss_threshold=0.001) 
    us.timer.toc('Training time')
    print('Training finished, loss:', dynamic_model_losses[-1])
    trainer.save_state(dynamic_model.flax_train_state, path=dyna_config.savedir)
    
    dynamic_model = ModelTrain(dyna_config)
    dynamic_model.flax_train_state, _ = trainer.load_state(dynamic_model.flax_train_state, path=dyna_config.savedir)

    test_ret = dynamic_model.test(dynamic_model.flax_train_state, dyna_train_xu[:10], jrng.new_key())
    print('Test result:', test_ret)
    print(dyna_train_dx[:10])
    # axs = us.plt.get_fig()  
    # axs[0].plot(dynamic_model_losses)
    # axs[0].set_yscale('log')
    # us.plt.show_pause()
    
if __name__ == '__main__':
    main()