from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action
import pandas as pd
import numpy as np
import joblib
import copy
import pkg_resources
import gym
from gym import spaces
from gym.utils import seeding
from datetime import datetime
from simglucose.analysis import reward_functions
from simglucose.analysis.risk import magni_risk_index
import random

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')


class T1DSimEnv(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, patient_name=None, reward_fun=None):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        self.source_dir = 'C:/Users/macke/PycharmProjects/simglucose/simglucose/params'
        if reward_fun == 'magni_reward':
            reward_fun = reward_functions.magni_reward
        seeds = self._seed()
        self.env = None
        nbr_hours = 1
        self.termination_penalty = 1e4
#        self.state_hist = int(1)
        self.state_hist = int((nbr_hours * 60) / 3)
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            patient_name = 'child#001'
        self.patient_name = patient_name
        patient = T1DPatient.withName(patient_name)
        sensor = CGMSensor.withName('Dexcom', seed=seeds[1])
        hour = self.np_random.randint(low=0.0, high=24.0)
        start_time = datetime(2018, 1, 1, hour, 0, 0)
        scenario = RandomScenario(start_time=start_time, seed=seeds[2])
        pump = InsulinPump.withName('Insulet')
        self.env = _T1DSimEnv(patient, sensor, pump, scenario)
        # TODO Set patient specific parmeters
        self.set_history_values(patient_name)
        self.reward_fun = reward_fun

    def step(self, action):
        # This gym only controls basal insulin
        # Universal Action Space
        action_scale = 'basal'
        basal_scaling = 43.2
        action_bias = 0
        if type(action) is np.ndarray:
            action = action.item()
        # 288 samples per day, bolus insulin should be 75% of insulin dose
        # split over 4 meals with 5 minute sampling rate, max unscaled value is 1+action_bias
        # https://care.diabetesjournals.org/content/34/5/1089
#        action = (action + action_bias) * ((self.ideal_basal * basal_scaling)/(1+action_bias))
        act = Action(basal=0, bolus=action)
        if self.reward_fun is None:
            _, reward, _, info =  self.env.step(act)
        else:
            _, reward, _, info = self.env.step(act, reward_fun=self.reward_fun)
        state = self.get_state()
        done = self.is_done()
        if done and self.termination_penalty is not None:
            reward = reward - self.termination_penalty
        return state, reward, done, info

    def is_done(self):
        return self.env.BG_hist[-1] < 40 or self.env.BG_hist[-1] > 400

    def reset(self):
        obs, _, _, _ = self.env.reset()
#        self._hist_init()
        return self.get_state()

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        seed3 = seeding.hash_seed(seed2 + 1) % 2**31
        return [seed1, seed2, seed3]

    def _render(self, mode='human', close=False):
        self.env.render(close=close)

    def get_state(self):
        bg = self.env.BG_hist[-self.state_hist:]
        bg = np.rint(bg)
        insulin = self.env.insulin_hist[-self.state_hist:]
        cho = self.env.CHO_hist[-self.state_hist:]
        if len(bg) < self.state_hist:
            bg = np.concatenate((np.full(self.state_hist - len(bg), -1), bg))
        if len(insulin) < self.state_hist:
            insulin = np.concatenate((np.full(self.state_hist - len(insulin), 0), insulin))
        if len(cho) < self.state_hist:
            cho = np.concatenate((np.full(self.state_hist - len(cho), 0), cho))

        return_arr = [bg, insulin]
        return np.stack(return_arr).flatten()
#        return_arr = np.array([bg])
#        return_arr = np.interp(return_arr, (return_arr.min(), return_arr.max()), (0, +1))
#        return np.stack(return_arr).flatten()

    def set_history_values(self, patient_name):
        self.patient_name = patient_name
        self.patient_para_file = '{}/vpatient_params.csv'.format(self.source_dir)
        self.control_quest = '{}/Quest2.csv'.format(self.source_dir)
        vpatient_params = pd.read_csv(self.patient_para_file)
        quest = pd.read_csv(self.control_quest)
        self.kind = self.patient_name.split('#')[0]
        self.bw = vpatient_params.query('Name=="{}"'.format(self.patient_name))['BW'].item()
        self.u2ss = vpatient_params.query('Name=="{}"'.format(self.patient_name))['u2ss'].item()
        self.ideal_basal = self.bw * self.u2ss / 6000.
        self.CR = quest.query('Name=="{}"'.format(patient_name)).CR.item()
        self.CF = quest.query('Name=="{}"'.format(patient_name)).CF.item()
        self.env_init_dict = joblib.load("{}/{}_data.pkl".format(self.source_dir, self.patient_name))
        self.env_init_dict['magni_risk_hist'] = []
        for bg in self.env_init_dict['bg_hist']:
            self.env_init_dict['magni_risk_hist'].append(magni_risk_index([bg]))
#        self._hist_init()

    def _hist_init(self):
        self.rolling = []
        env_init_dict = copy.deepcopy(self.env_init_dict)
        start_idx = random.randint(0,50)
        self.env.patient._state = env_init_dict['state']
        self.env.patient._t = env_init_dict['time']
        if self.start_date is not None:
            # need to reset date in start time
            orig_start_time = env_init_dict['time_hist'][0]
            new_start_time = datetime(year=self.start_date.year, month=self.start_date.month,
                                      day=self.start_date.day)
            new_time_hist = ((np.array(env_init_dict['time_hist']) - orig_start_time) + new_start_time).tolist()
            self.env.time_hist = new_time_hist
        else:
            self.env.time_hist = env_init_dict['time_hist']
        self.env.time_hist = env_init_dict['time_hist'][start_idx:start_idx+200]
        self.env.BG_hist = env_init_dict['bg_hist'][start_idx:start_idx+200]
        self.env.CGM_hist = env_init_dict['cgm_hist'][start_idx:start_idx+200]
        self.env.risk_hist = env_init_dict['risk_hist'][start_idx:start_idx+200]
        self.env.LBGI_hist = env_init_dict['lbgi_hist'][start_idx:start_idx+200]
        self.env.HBGI_hist = env_init_dict['hbgi_hist'][start_idx:start_idx+200]
        self.env.CHO_hist = env_init_dict['cho_hist'][start_idx:start_idx+200]
        self.env.insulin_hist = env_init_dict['insulin_hist'][start_idx:start_idx+200]
        self.env.magni_risk_hist = env_init_dict['magni_risk_hist'][start_idx:start_idx+200]
        self.env.time = self.env.time_hist[-1]

    def increment_seed(self, incr=1):
        self.seeds['numpy'] += incr
        self.seeds['scenario'] += incr
        self.seeds['sensor'] += incr

    @property
    def action_space(self):
#        ub = self.env.pump._params['max_basal']
        return spaces.Box(low=0, high=.4
                          , shape=(1,))

#    @property
#    def observation_space(self):
#        return spaces.Box(low=0, high=np.inf, shape=(1,))

    @property
    def observation_space(self):
        st = self.get_state()
#        num_channels = int(len(st)/self.state_hist)
#        return spaces.Box(low=0, high=np.inf, shape=(num_channels, self.state_hist))
        return spaces.Box(low=0, high=np.inf, shape=(len(st),))


