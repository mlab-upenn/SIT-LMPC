from flax.training import orbax_utils
import orbax
import numpy as np
import time
import os
import jax
            

class Trainer():
    def __init__(self, exp_name, savedir, 
                 max_epoch=300, best_fn=None, info_template=np.zeros(8),
                 initial_lr=1e-3, lr_method='step', rl_schedule_epoch=[], rl_schedule_gamma=0.5
                 ):
        self.exp_name = exp_name
        self.savedir = savedir
        self.max_epoch = max_epoch
        self.best_fn = best_fn
        self.orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        self.lr_scheduler = LR_scheduler(initial_lr, lr_method,
                                         rl_schedule_epoch, rl_schedule_gamma)
        self.best_info = np.ones_like(info_template) * np.inf
        self.epoch = 0
    
    def save_state(self, state, info=None, save_name='best', path=""):
        ckpt = {'state': state, 'info': info}
        save_args = orbax_utils.save_args_from_target(ckpt)
        self.orbax_checkpointer.save(path + self.exp_name + '_model_' + save_name, 
                                     ckpt, save_args=save_args, force=True)
    
    def load_state(self, state, info=None, save_name='best', path="", abs_path=False):
        ckpt = {'state': state, 'info': info}
        if abs_path:
            load_filename = path
        else:
            load_filename = self.savedir + self.exp_name + '_model_' + save_name
        state_restored = self.orbax_checkpointer.restore(load_filename, item=ckpt)
        print('Load from model: ', load_filename)
        return state_restored['state'], state_restored['info']
    
    def continue_train_load(self, state, info, path="", abs_path=False, 
                            transfer=False, transfer_exp_name=""):
        save_name = 'best'
        if abs_path:
            load_filename = path
        elif transfer:
            load_filename = path + transfer_exp_name + '_model_' + save_name
        else:
            load_filename = path + self.exp_name + '_model_' + save_name
        if os.path.exists(load_filename):
            state_loaded, info_loaded = \
                self.load_state(state, info, save_name, load_filename, abs_path=True)
            self.best_info = info_loaded
            for _ in range(int(info_loaded[0])):
                _ =  self.lr_scheduler.get_lr(self.epoch)
                self.step_epoch()
        return state_loaded
    
    def get_lr(self):
        return self.lr_scheduler.get_lr(self.epoch)
    
    def is_done(self):
        self.start_time = time.time()
        return self.epoch > self.max_epoch
    
    def step_epoch(self):
        self.epoch += 1
    
    def step(self, state, info):
        epoch_time = time.time() - self.start_time
        remaining_time = (self.max_epoch - self.epoch) * epoch_time / 3600
        return_text = f' |h_left {remaining_time:.1f}'
        
        self.save_state(state, info, 'last', self.savedir)
        if self.best_fn(self.best_info) > self.best_fn(info):
            self.best_info = info
            self.save_state(state, info, 'best', self.savedir)
            return_text += ' |BEST'
        self.epoch += 1
        return return_text
    


class LR_scheduler():
    def __init__(self, initial_lr, 
                 lr_method='step', 
                 rl_schedule_epoch=[], 
                 rl_schedule_gamma=0.5):
        self.method = lr_method
        self.schedule = rl_schedule_epoch
        self.gamma = rl_schedule_gamma
        self.next_scheduled_epoch = 0
        self.lr = initial_lr
        
    def get_lr(self, current_epoch):
        if self.method == 'step':
            return self.step_fn(current_epoch)
        
    def step_fn(self, current_epoch):
        if len(self.schedule) > 0:
            if current_epoch == self.schedule[self.next_scheduled_epoch]:
                if self.lr > self.lr * self.gamma:
                    self.lr = self.lr * self.gamma
                    self.next_scheduled_epoch += 1
                    print('Change LR to', self.lr)
        return self.lr
    



        

        
        
    
    