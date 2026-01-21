import jax
import os, time
import jax.numpy as jnp
import numpy as np
import utilsuite

class Plottings:
    def __init__(self, track, config, us, enable=True):
        self.enable = enable
        self.track = track
        self.config = config
        self.us = us
        
        self.waypoints = track.centerline.waypoints
        self.waypoints_render_subsample = self.waypoints[np.arange(self.waypoints.shape[0], step=self.waypoints.shape[0]//500), :] # render only 500 waypoints
        self.waypoints_boundary_l = jnp.concatenate([self.waypoints_render_subsample[:, 0:1], track.vmap_frenet_to_cartesian_jax(jnp.concatenate([self.waypoints_render_subsample[:, 0:1], 
                                                                                        jnp.ones_like(self.waypoints_render_subsample[:, 0:1]) * config.half_width,
                                                                                        jnp.zeros_like(self.waypoints_render_subsample[:, 0:1])], axis=1))], axis=1)
        self.waypoints_boundary_r = jnp.concatenate([self.waypoints_render_subsample[:, 0:1], track.vmap_frenet_to_cartesian_jax(jnp.concatenate([self.waypoints_render_subsample[:, 0:1], 
                                                                                            jnp.ones_like(self.waypoints_render_subsample[:, 0:1]) * -config.half_width,
                                                                                            jnp.zeros_like(self.waypoints_render_subsample[:, 0:1])], axis=1))], axis=1)
        self.timestamps = []
        self.start_time = None
        if not self.enable and not enable: return
        
    def plot_trajectory(self, key_option, laptime, enable=False):
        if not self.enable and not enable: return
        axs = self.us.plt.get_fig([1, 1])
        axs[0].scatter(self.waypoints_render_subsample[:, 1], self.waypoints_render_subsample[:, 2], s=1, c='grey')
        axs[0].scatter(self.waypoints_boundary_l[:, 1], self.waypoints_boundary_l[:, 2], s=1, c='k')
        axs[0].scatter(self.waypoints_boundary_r[:, 1], self.waypoints_boundary_r[:, 2], s=1, c='k')
        axs[0].scatter(self.us.rec.xy_lap_record[-1][:, 0], self.us.rec.xy_lap_record[-1][:, 1], s=1, c='r')
        axs[0].set_title(f'Trajectory {laptime:.2f}s')
        self.us.plt.save_fig(self.config.save_dir + 'trajectory')
        if key_option == '2': self.us.plt.show()
        self.us.plt.close_all()
        
    def plot_trajectory_history(self, lap_cnt, laptime, enable=False):
        if not self.enable and not enable: return
        axs = self.us.plt.get_fig([1, 1])
        axs[0].scatter(self.waypoints_render_subsample[:, 1], self.waypoints_render_subsample[:, 2], s=1, c='grey')
        axs[0].scatter(self.waypoints_boundary_l[:, 1], self.waypoints_boundary_l[:, 2], s=1, c='k')
        axs[0].scatter(self.waypoints_boundary_r[:, 1], self.waypoints_boundary_r[:, 2], s=1, c='k')
        axs[0].scatter(self.us.rec.xy_lap_record[-1][:, 0], self.us.rec.xy_lap_record[-1][:, 1], s=1, c='r')
        axs[0].set_title(f'Trajectory {laptime:.2f}s')
        if not os.path.exists(self.config.save_dir + 'traj_evo/'):
            os.makedirs(self.config.save_dir + 'traj_evo/')
        self.us.plt.save_fig(self.config.save_dir + 'traj_evo/' + 'trajectory_' + str(lap_cnt))
        self.us.plt.close_all()
        
    def plot_laptime_speed(self, key_option, laptime, enable=False):
        if not self.enable and not enable: return
        axs = self.us.plt.get_fig([2, 1], gridline=True)
        axs[0].plot(np.arange(len(self.us.rec.laptime_record)), self.us.rec.laptime_record)
        axs[0].set_title(f'Laptime vs. Iteration {laptime:.2f}s')
        axs[1].plot(np.arange(len(self.us.rec.mean_vels)), self.us.rec.mean_vels)
        axs[1].set_title(f'Average speed vs. Iteration {laptime:.2f}s')
        self.us.plt.save_fig(self.config.save_dir + 'laptime_speed')
        if key_option == '2':
            self.us.plt.show()
        self.us.plt.close_all()
            
    def plot_ss_violation(self, key_option, laptime, enable=False):
        if not self.enable and not enable: return
        axs = self.us.plt.get_fig([2, 1], gridline=True)
        axs[0].plot(np.arange(len(self.us.rec.ss_max_record)), self.us.rec.ss_max_record)
        axs[0].set_title(f'Max ss_violation vs. Iteration {laptime:.2f}s')
        axs[1].plot(np.arange(len(self.us.rec.ss_average_record)), self.us.rec.ss_average_record)
        axs[1].set_title(f'Average ss_violation vs. Iteration {laptime:.2f}s')
        self.us.plt.save_fig(self.config.save_dir + 'ss_violation')
        if key_option == '2':
            self.us.plt.show()
        self.us.plt.close_all()

    def plot_boundary_violation(self, key_option, laptime, enable=False):
        if not self.enable and not enable: return
        axs = self.us.plt.get_fig([2, 1], gridline=True)
        axs[0].plot(np.arange(len(self.us.rec.boundary_max_record)), self.us.rec.boundary_max_record)
        axs[0].set_title(f'Max boundary violation vs. Iteration {laptime:.2f}s')
        axs[1].plot(np.arange(len(self.us.rec.boundary_average_record)), self.us.rec.boundary_average_record)
        axs[1].set_title(f'Average boundary violation vs. Iteration {laptime:.2f}s')
        self.us.plt.save_fig(self.config.save_dir + 'boundary_violation')
        if key_option == '2':
            self.us.plt.show()
        self.us.plt.close_all()
    
    def plot_speed_position(self, key_option, laptime, enable=False):
        if not self.enable and not enable: return
        axs = self.us.plt.get_fig([1, 1])
        self.us.plt.background((0, 0, 0), 0)
        axs[0].scatter(self.us.rec.xy_lap_record[-1][:, 0], self.us.rec.xy_lap_record[-1][:, 1], s=2, c=self.us.rec.xy_lap_record[-1][:, 3], cmap='coolwarm')
        axs[0].set_title(f'Speed vs. Position {laptime:.2f}s')
        self.us.plt.colorbar(0, cmap='coolwarm')
        axs[0].scatter(self.waypoints_boundary_l[:, 1], self.waypoints_boundary_l[:, 2], s=1, c='w')
        axs[0].scatter(self.waypoints_boundary_r[:, 1], self.waypoints_boundary_r[:, 2], s=1, c='w')
        self.us.plt.save_fig(self.config.save_dir + 'speed_position')
        if key_option == '2':
            self.us.plt.show()
        self.us.plt.close_all()
        
    def plot_min_speed_position(self, key_option, laptime, enable=False):
        if not self.enable and not enable: return
        axs = self.us.plt.get_fig([1, 1])
        self.us.plt.background((0, 0, 0), 0)
        axs[0].scatter(self.us.rec.min_states[:, 0], self.us.rec.min_states[:, 1], s=2, c=self.us.rec.min_states[:, 3], cmap='coolwarm')
        axs[0].set_title(f'Speed vs. Position {laptime:.2f}s')
        self.us.plt.colorbar(0, cmap='coolwarm')
        axs[0].scatter(self.waypoints_boundary_l[:, 1], self.waypoints_boundary_l[:, 2], s=1, c='w')
        axs[0].scatter(self.waypoints_boundary_r[:, 1], self.waypoints_boundary_r[:, 2], s=1, c='w')
        self.us.plt.save_fig(self.config.save_dir + 'speed_position')
        if key_option == '2':
            self.us.plt.show()
        self.us.plt.close_all()


    def plot_speed_position_history(self, lap_cnt, laptime, enable=False):
        if not self.enable and not enable: return
        self.timestamps.append(time.time() - self.start_time)
        self.start_time = time.time()
        axs = self.us.plt.get_fig([1, 1])
        self.us.plt.background((0, 0, 0), 0)
        axs[0].scatter(self.us.rec.xy_states[:, 0], self.us.rec.xy_states[:, 1], s=2, c=self.us.rec.xy_states[:, 3], cmap='coolwarm')
        axs[0].set_title(f'Speed vs. Position {laptime:.2f}s' + ', Lap ' + str(lap_cnt))
        self.us.plt.colorbar(0, cmap='coolwarm')
        axs[0].scatter(self.waypoints_boundary_l[:, 1], self.waypoints_boundary_l[:, 2], s=1, c='w')
        axs[0].scatter(self.waypoints_boundary_r[:, 1], self.waypoints_boundary_r[:, 2], s=1, c='w')
        if not os.path.exists(self.config.save_dir + 'speed_evo/'):
            os.makedirs(self.config.save_dir + 'speed_evo/')
        # self.us.plt.plt.show(block=False)
        self.us.plt.save_fig(self.config.save_dir + 'speed_evo/' + 'speed_position_' + str(lap_cnt))
        np.savetxt(self.config.save_dir + 'speed_evo/' + "timestamps.txt", self.timestamps)
        self.us.plt.close_all()
        
    def plot_acceleration_position(self, key_option, laptime, enable=False):
        if not self.enable and not enable: return
        axs = self.us.plt.get_fig([1, 1])
        self.us.plt.background((0, 0, 0), 0)
        axs[0].scatter(self.us.rec.xy_lap_record[-1][:, 0], self.us.rec.xy_lap_record[-1][:, 1], s=2, c=self.us.rec.control_records[-1][:, 0], cmap='coolwarm')
        axs[0].set_title(f'Acceleration vs. Position {laptime:.2f}s')
        self.us.plt.colorbar(0, cmap='coolwarm')
        axs[0].scatter(self.waypoints_boundary_l[:, 1], self.waypoints_boundary_l[:, 2], s=1, c='w')
        axs[0].scatter(self.waypoints_boundary_r[:, 1], self.waypoints_boundary_r[:, 2], s=1, c='w')
        self.us.plt.save_fig(self.config.save_dir + 'acceleration_position')
        if key_option == '2':
            self.us.plt.show()
        self.us.plt.close_all()
        
    def plot_acceleration_position_history(self, lap_cnt, laptime, enable=False):
        if not self.enable and not enable: return
        axs = self.us.plt.get_fig([1, 1])
        self.us.plt.background((0, 0, 0), 0)
        axs[0].scatter(self.us.rec.xy_lap_record[-1][:, 0], self.us.rec.xy_lap_record[-1][:, 1], s=2, c=self.us.rec.control_records[-1][:, 0], cmap='coolwarm')
        axs[0].set_title(f'Acceleration vs. Position {laptime:.2f}s')
        self.us.plt.colorbar(0, cmap='coolwarm')
        axs[0].scatter(self.waypoints_boundary_l[:, 1], self.waypoints_boundary_l[:, 2], s=1, c='w')
        axs[0].scatter(self.waypoints_boundary_r[:, 1], self.waypoints_boundary_r[:, 2], s=1, c='w')
        if not os.path.exists(self.config.save_dir + 'accel_evo/'):
            os.makedirs(self.config.save_dir + 'accel_evo/')
        self.us.plt.save_fig(self.config.save_dir + 'accel_evo/' + 'acceleration_position_' + str(lap_cnt))
        self.us.plt.close_all()
        
    def plot_ss_lamb_history(self, lap_cnt, laptime, enable=False):
        if not self.enable and not enable: return
        axs = self.us.plt.get_fig([1, 1])
        self.us.plt.background((0, 0, 0), 0)
        axs[0].scatter(self.us.rec.xy_lap_record[-1][:, 0], self.us.rec.xy_lap_record[-1][:, 1], s=2, 
                       c=np.array(self.us.rec.lamb_record)[:, 0], cmap='coolwarm')
        axs[0].set_title(f'Safeset Lambs vs. Position {laptime:.2f}s')
        self.us.plt.colorbar(0, cmap='coolwarm')
        axs[0].scatter(self.waypoints_boundary_l[:, 1], self.waypoints_boundary_l[:, 2], s=1, c='w')
        axs[0].scatter(self.waypoints_boundary_r[:, 1], self.waypoints_boundary_r[:, 2], s=1, c='w')
        if not os.path.exists(self.config.save_dir + 'sslamb_evo/'):
            os.makedirs(self.config.save_dir + 'sslamb_evo/')
        self.us.plt.save_fig(self.config.save_dir + 'sslamb_evo/' + 'ss_lamb_position_' + str(lap_cnt))
        self.us.plt.close_all()
        
    def plot_boundary_lamb_history(self, lap_cnt, laptime, enable=False):
        if not self.enable and not enable: return
        axs = self.us.plt.get_fig([1, 1])
        self.us.plt.background((0, 0, 0), 0)
        axs[0].scatter(self.us.rec.xy_lap_record[-1][:, 0], self.us.rec.xy_lap_record[-1][:, 1], s=2, 
                       c=np.array(self.us.rec.lamb_record)[:, 1], cmap='coolwarm')
        axs[0].set_title(f'Boundary Lambs vs. Position {laptime:.2f}s')
        self.us.plt.colorbar(0, cmap='coolwarm')
        axs[0].scatter(self.waypoints_boundary_l[:, 1], self.waypoints_boundary_l[:, 2], s=1, c='w')
        axs[0].scatter(self.waypoints_boundary_r[:, 1], self.waypoints_boundary_r[:, 2], s=1, c='w')
        if not os.path.exists(self.config.save_dir + 'boundarylamb_evo/'):
            os.makedirs(self.config.save_dir + 'boundarylamb_evo/')
        self.us.plt.save_fig(self.config.save_dir + 'boundarylamb_evo/' + 'boundary_lamb_position_' + str(lap_cnt))
        self.us.plt.close_all()
        
    def plot_lap_lambs(self, lap_cnt, laptime, enable=False):
        if not self.enable and not enable: return
        axs = self.us.plt.get_fig([2, 1], gridline=True)
        axs[0].plot(np.arange(len(self.us.rec.laptime_record)), np.array(self.us.rec.mean_lambs)[:, 0])
        axs[0].set_title(f'Average Safeset Lambs {laptime:.2f}s')
        axs[1].plot(np.arange(len(self.us.rec.laptime_record)), np.array(self.us.rec.mean_lambs)[:, 1])
        axs[1].set_title(f'Average Boundary Lambs {laptime:.2f}s')
        self.us.plt.save_fig(self.config.save_dir + 'lap_lambs')
        self.us.plt.close_all()

class LMPPI_Plottings:
    def __init__(self, config, enable=True):
        self.enable = enable
        if not self.enable and not enable: return
        self.plt_utils = utilsuite.pltUtils()
        self.config = config
        self.save_cnt = 0
        
    def plot_rewards_adaptive_method(self, ref_ss_s, states, 
                     traj_opt, r_value, reward, r_sshull, r_boundary, half_width,
                     s_opt_ss_distance, boundary_distance, iter_cnt, show=False, enable=True):
        if not self.enable and not enable: return
        sampled_states_frenet_np = np.array(jax.device_get(states))
        sampled_states_frenet_np_last = sampled_states_frenet_np[:, -1]
        sampled_states_frenet_np = sampled_states_frenet_np.reshape(-1, 7)
        axs = self.plt_utils.get_fig([1, 4], figsize=[16, 6])
        
        axs[0].scatter(-sampled_states_frenet_np_last[:, 1], sampled_states_frenet_np_last[:, 0], c=reward[:, -1].reshape(-1), s=1)
        axs[0].set_title(f'reward_func')
        self.plt_utils.colorbar(0)
        axs[0].plot(-ref_ss_s[:, 1], ref_ss_s[:, 0], 'b.')
        axs[0].scatter(-sampled_states_frenet_np_last[np.where(np.abs(sampled_states_frenet_np_last[:, 1]) > half_width)][:, 1], 
                    sampled_states_frenet_np_last[np.where(np.abs(sampled_states_frenet_np_last[:, 1]) > half_width)][:, 0], s=0.5, c='k')
        axs[0].plot(-traj_opt[:, 1], traj_opt[:, 0], '.r')

        if len(r_value.reshape(-1)) < len(sampled_states_frenet_np[:, 1]):
            axs[1].scatter(-sampled_states_frenet_np_last[:, 1], sampled_states_frenet_np_last[:, 0], c=r_value[:, -1].reshape(-1), s=1)
        else:
            axs[1].scatter(-sampled_states_frenet_np[:, 1], sampled_states_frenet_np[:, 0], c=r_value.reshape(-1), s=1)
        axs[1].set_title(f'value_func')
        self.plt_utils.colorbar(1)

        # axs[2].scatter(-sampled_states_frenet_np_last[:, 1], sampled_states_frenet_np_last[:, 0], c=r_sshull[:, -1].reshape(-1), s=1)
        axs[2].scatter(-sampled_states_frenet_np[:, 1], sampled_states_frenet_np[:, 0], c=r_sshull.reshape(-1), s=1)
        axs[2].set_title(f'r_sshull')
        self.plt_utils.colorbar(2)
        axs[3].scatter(-sampled_states_frenet_np[:, 1], sampled_states_frenet_np[:, 0], c=r_boundary.reshape(-1), s=1)
        axs[3].set_title(f'r_boundary')
        self.plt_utils.plt.suptitle(f'iter_cnt: {iter_cnt}, s_opt_ss_distance: {s_opt_ss_distance}, boundary_distance: {boundary_distance}')
        self.plt_utils.colorbar(3)
        
        if not os.path.exists(self.config.save_dir + 'debug/'):
            os.makedirs(self.config.save_dir + 'debug/')
        
        self.save_cnt += 1
        if show:
            self.plt_utils.save_fig(self.config.save_dir + 'debug/' + str(self.save_cnt))
            self.plt_utils.show_pause()
        self.plt_utils.close_all()
        
    def plot_rewards_point(self, ref_ss_s, states, 
                    traj_opt, lamb_ind, lambs_opt, r_value, reward, r_sshull, r_boundary, half_width,
                    s_opt_ss_distance, boundary_distance, enable=True, show=False):
        if not self.enable and not enable: return
        sampled_states_frenet_np = np.array(jax.device_get(states))
        sampled_states_frenet_np_last = sampled_states_frenet_np[:, -1]
        sampled_states_frenet_np = sampled_states_frenet_np.reshape(-1, 7)
        axs = self.plt_utils.get_fig([1, 5], figsize=[16, 6])
        
        axs[0].scatter(sampled_states_frenet_np_last[:, 0], sampled_states_frenet_np_last[:, 1], 
                       c=reward[lamb_ind, :, -1].reshape(-1), s=1)
        axs[0].set_title(f'reward_func')
        self.plt_utils.colorbar(0)
        axs[0].plot(ref_ss_s[:, 0], ref_ss_s[:, 1], 'b.')
        # axs[0].scatter(-sampled_states_frenet_np_last[np.where(np.abs(sampled_states_frenet_np_last[:, 1]) > half_width)][:, 1], 
        #             sampled_states_frenet_np_last[np.where(np.abs(sampled_states_frenet_np_last[:, 1]) > half_width)][:, 0], s=0.5, c='k')
        
        traj_opt_others = np.delete(traj_opt, lamb_ind, axis=0)
        axs[0].plot(traj_opt_others[:, :, 0], traj_opt_others[:, :, 1], '.g')
        axs[0].plot(traj_opt[lamb_ind, :, 0], traj_opt[lamb_ind, :, 1], '.r')
        
        
        # axs[0].scatter(-sampled_states_frenet_np[:, 1], sampled_states_frenet_np[:, 0], c=reward[lamb_ind].reshape(-1))
        # axs[0].set_title(f'reward_func')
        # self.plt_utils.colorbar(0)
        # axs[0].plot(-ref_ss_s[:, 1], ref_ss_s[:, 0], 'b.')
        # axs[0].scatter(-sampled_states_frenet_np[np.where(np.abs(sampled_states_frenet_np[:, 1]) > half_width)][:, 1], 
        #             sampled_states_frenet_np[np.where(np.abs(sampled_states_frenet_np[:, 1]) > half_width)][:, 0], s=0.5, c='k')
        # axs[0].plot(-traj_opt[lamb_ind, :, 1], traj_opt[lamb_ind, :, 0], '.r')
        
        # if len(r_value.reshape(-1)) < len(sampled_states_frenet_np[:, 1]):
        axs[1].scatter(sampled_states_frenet_np_last[:, 0], sampled_states_frenet_np_last[:, 1], c=r_value[:, -1].reshape(-1), s=1)
        # else:
        #     axs[1].scatter(-sampled_states_frenet_np[:, 1], sampled_states_frenet_np[:, 0], c=r_value.reshape(-1), s=1)
        axs[1].set_title(f'value_func')
        self.plt_utils.colorbar(1)
        
        axs[2].scatter(sampled_states_frenet_np_last[:, 0], sampled_states_frenet_np_last[:, 1], c=r_sshull[:, -1].reshape(-1), s=1)
        # axs[2].scatter(-sampled_states_frenet_np[:, 1], sampled_states_frenet_np[:, 0], c=-r_sshull.reshape(-1))
        axs[2].plot(ref_ss_s[:, 0], ref_ss_s[:, 1], 'b.')
        axs[2].set_title(f'r_sshull')
        self.plt_utils.colorbar(2)
        axs[3].scatter(sampled_states_frenet_np[:, 0], sampled_states_frenet_np[:, 1], c=r_boundary.reshape(-1), s=1)
        axs[3].set_title(f'r_boundary')
        self.plt_utils.plt.suptitle(f'lambs_opt: {lambs_opt}, s_opt_ss_distance: {s_opt_ss_distance}, boundary_distance: {boundary_distance}')
        self.plt_utils.colorbar(3)
        
        if not os.path.exists(self.config.save_dir + 'debug/'):
            os.makedirs(self.config.save_dir + 'debug/')
        
        self.save_cnt += 1
        if show:
            # self.plt_utils.save_fig(self.config.save_dir + 'debug/' + str(self.save_cnt))
            # self.plt_utils.show()
            self.plt_utils.show_pause()
        self.plt_utils.close_all()
        
    def plot_rewards(self, ref_ss_s, states, 
                     traj_opt, lamb_ind, lambs_opt, r_value, sampled_value_var, reward, r_sshull, r_boundary, half_width,
                     s_opt_ss_distance, boundary_distance, enable=True, show=False):
        if not self.enable and not enable: return
        sampled_states_frenet_np = np.array(jax.device_get(states))
        sampled_states_frenet_np_last = sampled_states_frenet_np[:, -1]
        sampled_states_frenet_np = sampled_states_frenet_np.reshape(-1, 7)
        axs = self.plt_utils.get_fig([1, 5], figsize=[16, 6])
        
        axs[0].scatter(-sampled_states_frenet_np_last[:, 1], sampled_states_frenet_np_last[:, 0], 
                       c=reward[lamb_ind, :, -1].reshape(-1), s=1)
        axs[0].set_title(f'reward_func')
        self.plt_utils.colorbar(0)
        axs[0].plot(-ref_ss_s[:, 1], ref_ss_s[:, 0], 'b.')
        axs[0].scatter(-sampled_states_frenet_np_last[np.where(np.abs(sampled_states_frenet_np_last[:, 1]) > half_width)][:, 1], 
                    sampled_states_frenet_np_last[np.where(np.abs(sampled_states_frenet_np_last[:, 1]) > half_width)][:, 0], s=0.5, c='k')
        
        traj_opt_others = np.delete(traj_opt, lamb_ind, axis=0)
        axs[0].plot(-traj_opt_others[:, :, 1], traj_opt_others[:, :, 0], '.g')
        axs[0].plot(-traj_opt[lamb_ind, :, 1], traj_opt[lamb_ind, :, 0], '.r')
        
        
        # axs[0].scatter(-sampled_states_frenet_np[:, 1], sampled_states_frenet_np[:, 0], c=reward[lamb_ind].reshape(-1))
        # axs[0].set_title(f'reward_func')
        # self.plt_utils.colorbar(0)
        # axs[0].plot(-ref_ss_s[:, 1], ref_ss_s[:, 0], 'b.')
        # axs[0].scatter(-sampled_states_frenet_np[np.where(np.abs(sampled_states_frenet_np[:, 1]) > half_width)][:, 1], 
        #             sampled_states_frenet_np[np.where(np.abs(sampled_states_frenet_np[:, 1]) > half_width)][:, 0], s=0.5, c='k')
        # axs[0].plot(-traj_opt[lamb_ind, :, 1], traj_opt[lamb_ind, :, 0], '.r')
        
        if len(r_value.reshape(-1)) < len(sampled_states_frenet_np[:, 1]):
            axs[1].scatter(-sampled_states_frenet_np_last[:, 1], sampled_states_frenet_np_last[:, 0], c=r_value[:, -1].reshape(-1), s=1)
        else:
            axs[1].scatter(-sampled_states_frenet_np[:, 1], sampled_states_frenet_np[:, 0], c=r_value.reshape(-1), s=1)
        axs[1].set_title(f'value_func')
        self.plt_utils.colorbar(1)
        
        if len(r_value.reshape(-1)) < len(sampled_states_frenet_np[:, 1]):
            axs[2].scatter(-sampled_states_frenet_np_last[:, 1], sampled_states_frenet_np_last[:, 0], c=sampled_value_var[:, -1].reshape(-1), s=1)
        else:
            axs[2].scatter(-sampled_states_frenet_np[:, 1], sampled_states_frenet_np[:, 0], c=sampled_value_var.reshape(-1), s=1)
        axs[2].set_title(f'value_func_var')
        self.plt_utils.colorbar(2)

        axs[3].scatter(-sampled_states_frenet_np_last[:, 1], sampled_states_frenet_np_last[:, 0], c=r_sshull[:, -1].reshape(-1), s=1)
        # axs[2].scatter(-sampled_states_frenet_np[:, 1], sampled_states_frenet_np[:, 0], c=-r_sshull.reshape(-1))
        axs[3].plot(-ref_ss_s[:, 1], ref_ss_s[:, 0], 'b.')
        axs[3].set_title(f'r_sshull')
        self.plt_utils.colorbar(3)
        axs[4].scatter(-sampled_states_frenet_np[:, 1], sampled_states_frenet_np[:, 0], c=-r_boundary.reshape(-1), s=1)
        axs[4].set_title(f'r_boundary')
        self.plt_utils.plt.suptitle(f'lambs_opt: {lambs_opt}, s_opt_ss_distance: {s_opt_ss_distance}, boundary_distance: {boundary_distance}')
        self.plt_utils.colorbar(4)
        
        if not os.path.exists(self.config.save_dir + 'debug/'):
            os.makedirs(self.config.save_dir + 'debug/')
        
        self.save_cnt += 1
        if show:
            # self.plt_utils.save_fig(self.config.save_dir + 'debug/' + str(self.save_cnt))
            # self.plt_utils.show()
            self.plt_utils.show_pause()
        self.plt_utils.close_all()
        
    def plot_rewards_and_sample_point(self, ref_ss_s, states, 
                     traj_opt, lamb_ind, lambs_opt, r_value, sampled_value_var, reward, r_sshull, r_boundary, half_width,
                     s_opt_ss_distance, boundary_distance, sampled_actions, prev_a_opt, a_opt, enable=True, show=False):
        if not self.enable and not enable: return
        sampled_states_frenet_np = np.array(jax.device_get(states))
        sampled_states_frenet_np_last = sampled_states_frenet_np[:, -1]
        sampled_states_frenet_np = sampled_states_frenet_np.reshape(-1, 7)
        axs = self.plt_utils.get_fig([1, 6], figsize=[16, 6])
        
        axs[0].scatter(-sampled_states_frenet_np_last[:, 1], sampled_states_frenet_np_last[:, 0], 
                       c=reward[lamb_ind, :, -1].reshape(-1), s=1)
        axs[0].set_title(f'reward_func')
        self.plt_utils.colorbar(0)
        axs[0].plot(-ref_ss_s[:, 1], ref_ss_s[:, 0], 'b.')
        # axs[0].scatter(-sampled_states_frenet_np_last[np.where(np.abs(sampled_states_frenet_np_last[:, 1]) > half_width)][:, 1], 
        #             sampled_states_frenet_np_last[np.where(np.abs(sampled_states_frenet_np_last[:, 1]) > half_width)][:, 0], s=0.5, c='k')
        
        traj_opt_others = np.delete(traj_opt, lamb_ind, axis=0)
        # axs[0].plot(-traj_opt_others[:, :, 1], traj_opt_others[:, :, 0], '.g')
        axs[0].plot(-traj_opt[lamb_ind, -1, 1], traj_opt[lamb_ind, -1, 0], '.r')
        
        if len(r_value.reshape(-1)) < len(sampled_states_frenet_np[:, 1]):
            axs[1].scatter(-sampled_states_frenet_np_last[:, 1], sampled_states_frenet_np_last[:, 0], c=r_value[:, -1].reshape(-1), s=1)
        else:
            axs[1].scatter(-sampled_states_frenet_np[:, 1], sampled_states_frenet_np[:, 0], c=r_value.reshape(-1), s=1)
        axs[1].set_title(f'value_func')
        self.plt_utils.colorbar(1)

        axs[3].scatter(-sampled_states_frenet_np_last[:, 1], sampled_states_frenet_np_last[:, 0], c=r_sshull[:, -1].reshape(-1), s=1)
        # axs[2].scatter(-sampled_states_frenet_np[:, 1], sampled_states_frenet_np[:, 0], c=-r_sshull.reshape(-1))
        axs[3].plot(-ref_ss_s[:, 1], ref_ss_s[:, 0], 'b.')
        axs[3].set_title(f'r_sshull')
        self.plt_utils.colorbar(3)
        axs[4].scatter(-sampled_states_frenet_np[:, 1], sampled_states_frenet_np[:, 0], c=-r_boundary.reshape(-1), s=1)
        axs[4].set_title(f'r_boundary')
        self.plt_utils.plt.suptitle(f'lambs_opt: {lambs_opt}, s_opt_ss_distance: {s_opt_ss_distance:.5f}, ' +
                                    f'boundary_distance: {boundary_distance:.5f}, reward: {reward[lamb_ind, :, -1].mean():.5f}')
        self.plt_utils.colorbar(4)
        
        for ind in range(sampled_actions.shape[1]):
            axs[5].scatter(sampled_actions[:, ind, 0], sampled_actions[:, ind, 1] + 3*ind, c='b', s=1)
            axs[5].scatter(prev_a_opt[ind, 0], prev_a_opt[ind, 1] + 3*ind, c='r', s=2)
            axs[5].scatter(a_opt[ind, 0], a_opt[ind, 1] + 3*ind, c='y', s=2)
            
        
        if not os.path.exists(self.config.save_dir + 'debug/'):
            os.makedirs(self.config.save_dir + 'debug/')
        
        self.save_cnt += 1
        if show:
            # self.plt_utils.save_fig(self.config.save_dir + 'debug/' + str(self.save_cnt))
            self.plt_utils.show()
            # self.plt_utils.show_pause()
        self.plt_utils.close_all()
        
    def plot_rewards_and_sample(self, iter_ind, ref_ss_s, states, 
                     traj_opt, lamb_ind, lambs_opt, r_value, boundary_distance_trajs, reward, r_sshull, r_boundary, half_width,
                     s_opt_ss_distance, boundary_distance, sampled_actions, prev_a_opt, a_opt, enable=True, show=False):
        if not self.enable and not enable: return
        sampled_states_frenet_np = jax.device_get(states)
        sampled_states_frenet_np_last = sampled_states_frenet_np[:, -1]
        sampled_states_frenet_np = sampled_states_frenet_np.reshape(-1, 7)
        axs = self.plt_utils.get_fig([1, 6], figsize=[16, 6])
        
        axs[0].scatter(-sampled_states_frenet_np_last[:, 1], sampled_states_frenet_np_last[:, 0], 
                       c=reward[lamb_ind, :, -1].reshape(-1), s=1)
        axs[0].set_title(f'reward_func')
        self.plt_utils.colorbar(0)
        axs[0].plot(-ref_ss_s[:, 1], ref_ss_s[:, 0], 'b.')
        # axs[0].scatter(-sampled_states_frenet_np_last[np.where(np.abs(sampled_states_frenet_np_last[:, 1]) > half_width)][:, 1], 
        #             sampled_states_frenet_np_last[np.where(np.abs(sampled_states_frenet_np_last[:, 1]) > half_width)][:, 0], s=0.5, c='k')
        
        # if lamb_ind != -1:
        #     traj_opt_others = np.delete(traj_opt, lamb_ind, axis=0)
        #     axs[0].plot(-traj_opt_others[:, :, 1], traj_opt_others[:, :, 0], '.g')
        axs[0].plot(-traj_opt[lamb_ind, :, 1], traj_opt[lamb_ind, :, 0], '.r')
        
        axs[1].scatter(-sampled_states_frenet_np_last[:, 1], sampled_states_frenet_np_last[:, 0], c=r_value[:, -1].reshape(-1), s=1)
        axs[1].set_title(f'value_func')
        self.plt_utils.colorbar(1)
        
        axs[2].scatter(-traj_opt.reshape(-1, 7)[:, 1], traj_opt.reshape(-1, 7)[:, 0], c=boundary_distance_trajs.reshape(-1), s=1)
        axs[2].set_title(f'boundary_distance_trajs')
        self.plt_utils.colorbar(2)

        axs[3].scatter(-sampled_states_frenet_np_last[:, 1], sampled_states_frenet_np_last[:, 0], c=r_sshull[:, -1].reshape(-1), s=1)
        # axs[2].scatter(-sampled_states_frenet_np[:, 1], sampled_states_frenet_np[:, 0], c=-r_sshull.reshape(-1))
        axs[3].plot(-ref_ss_s[:, 1], ref_ss_s[:, 0], 'b.')
        axs[3].set_title(f'r_sshull')
        self.plt_utils.colorbar(3)
        axs[4].scatter(-sampled_states_frenet_np[:, 1], sampled_states_frenet_np[:, 0], c=-r_boundary.reshape(-1), s=1)
        axs[4].plot(-traj_opt[lamb_ind, :, 1], traj_opt[lamb_ind, :, 0], '.r', markersize=1)
        axs[4].set_title(f'r_boundary')
        self.plt_utils.plt.suptitle(f'lambs_opt: {lambs_opt}, s_opt_ss_distance: {s_opt_ss_distance:.5f}, ' +
                                    f'boundary_distance: {boundary_distance:.5f}, reward: {reward[lamb_ind, :, -1].mean():.5f}, iter_ind: {iter_ind}')
        self.plt_utils.colorbar(4)
        
        for ind in range(sampled_actions.shape[1]):
            axs[5].scatter(sampled_actions[:, ind, 0], sampled_actions[:, ind, 1] + 3*ind, c='b', s=1)
            axs[5].scatter(prev_a_opt[ind, 0], prev_a_opt[ind, 1] + 3*ind, c='y', s=3)
            axs[5].scatter(a_opt[ind, 0], a_opt[ind, 1] + 3*ind, c='r', s=3)
            
        
        if not os.path.exists(self.config.save_dir + 'debug/'):
            os.makedirs(self.config.save_dir + 'debug/')
        
        self.save_cnt += 1
        if show:
            # self.plt_utils.save_fig(self.config.save_dir + 'debug/' + str(self.save_cnt))
            # self.plt_utils.show()
            self.plt_utils.show_pause()
        self.plt_utils.close_all()
        
    def plot_distance(self, s_opt_ss_distance, boundary_distance, show):
        if not self.enable and not enable: return
        axs = self.plt_utils.get_fig([2, 1], figsize=[16, 6])
        axs[0].plot(s_opt_ss_distance, '.')
        axs[0].set_title(f's_opt_ss_distance')
        axs[1].plot(boundary_distance, '.')
        axs[1].set_title(f'boundary_distance')
        if show:
            self.plt_utils.show_pause()
        self.plt_utils.close_all()
    # zeros_arr = np.zeros_like(self.lambs[:, 0][feasible_list])
    # axs = self.plt_utils.get_fig([1, 1])
    # axs[0].scatter(self.lambs[:, 0], self.lambs[:, 1], c=s_opt_ss_distance)
    # axs[0].scatter(self.lambs[:, 0][feasible_list], self.lambs[:, 1][feasible_list], c=zeros_arr)
    # axs[0].set_title(f's_opt_ss_distance')
    # axs[0].set_xlabel('ss_lambda')
    # axs[0].set_ylabel('boundary_lambda')
    # self.plt_utils.save_fig(self.config.save_dir + 's_opt_ss_distance')
    # self.plt_utils.colorbar(0)
    # self.plt_utils.show_pause()


    # axs = self.plt_utils.get_fig([1, 1])
    # axs[0].scatter(self.lambs[:, 0], self.lambs[:, 1], c=boundary_distance)
    # axs[0].scatter(self.lambs[:, 0][feasible_list], self.lambs[:, 1][feasible_list], c=zeros_arr)
    # axs[0].set_title(f'boundary_distance')
    # self.plt_utils.save_fig(self.config.save_dir + 'boundary_distance')
    # self.plt_utils.colorbar(0)
    # self.plt_utils.show_pause()

    # axs = self.plt_utils.get_fig([1, 1])
    # axs[0].scatter(self.lambs[:, 0], self.lambs[:, 1], c=samples_s_opt_save)
    # infeasible_list = np.where((s_opt_ss_distance >= ss_relaxation) | (boundary_distance != 0))[0]
    # axs[0].scatter(self.lambs[:, 0][infeasible_list], self.lambs[:, 1][infeasible_list], c='k')
    # axs[0].set_title(f'samples_s_opt_save')
    # self.plt_utils.save_fig(self.config.save_dir + 'samples_s_opt_save')
    # self.plt_utils.colorbar(0)


    # inds = np.random.choice(sampled_states_frenet_np.shape[0], 2000, replace=False)  
    # sampled_states_frenet_np = sampled_states_frenet_np[inds]
    # axs = self.plt_utils.get_fig([1, 4])
    # axs[0].scatter(-sampled_states_frenet_np[:, 1], sampled_states_frenet_np[:, 0], c=r_value.reshape(-1)[inds])
    # axs[0].set_title(f'value_func')
    # self.plt_utils.colorbar(0)

    # axs[1].scatter(-sampled_states_frenet_np[:, 1], sampled_states_frenet_np[:, 0], c=reward[lamb_ind].reshape(-1)[inds])
    # axs[1].set_title(f'reward_func')
    # self.plt_utils.colorbar(1)
    # axs[1].plot(-ref_ss_s[:, 1], ref_ss_s[:, 0], 'b.')
    # axs[1].scatter(-sampled_states_frenet_np[np.where(np.abs(sampled_states_frenet_np[:, 1]) > self.half_width)][:, 1], 
    #             sampled_states_frenet_np[np.where(np.abs(sampled_states_frenet_np[:, 1]) > self.half_width)][:, 0], s=0.5, c='k')
    # axs[1].plot(-s_opt_frenet[lamb_ind, :, 1], s_opt_frenet[lamb_ind, :, 0], '.r')

    # axs[2].scatter(-sampled_states_frenet_np[:, 1], sampled_states_frenet_np[:, 0], c=r_sshull.reshape(-1)[inds])
    # axs[2].set_title(f'r_sshull')
    # self.plt_utils.colorbar(2)
    # axs[3].scatter(-sampled_states_frenet_np[:, 1], sampled_states_frenet_np[:, 0], c=r_boundary.reshape(-1)[inds])
    # axs[3].set_title(f'r_boundary')
    # self.plt_utils.colorbar(3)