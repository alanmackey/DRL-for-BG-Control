from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action
import numpy as np
import pkg_resources
import gym
from gym import spaces
from gym.utils import seeding
from datetime import datetime

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
        seeds = self._seed()
        nbr_hours = 4
        self.termination_penalty = 5
        self.state_hist = int((nbr_hours * 60) / 3)
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            patient_name = 'adolescent#001'
        patient = T1DPatient.withName(patient_name)
        sensor = CGMSensor.withName('Dexcom', seed=seeds[1])
        hour = self.np_random.randint(low=0.0, high=24.0)
        start_time = datetime(2018, 1, 1, hour, 0, 0)
        scenario = RandomScenario(start_time=start_time, seed=seeds[2])
        pump = InsulinPump.withName('Insulet')
        self.env = _T1DSimEnv(patient, sensor, pump, scenario)
        self.reward_fun = reward_fun

    def _step(self, action):
        # This gym only controls basal insulin
        if type(action) is np.ndarray:
            action = action.item()
        act = Action(basal=action, bolus=0)
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
        return self.env.BG_hist[-1] < 40 or self.env.BG_hist[-1] > 190

    def _reset(self):
        obs, _, _, _ = self.env.reset()
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
        bg = self.env.CGM_hist[-self.state_hist:]
        insulin = self.env.insulin_hist[-self.state_hist:]
        if len(bg) < self.state_hist:
            bg = np.concatenate((np.full(self.state_hist - len(bg), -1), bg))
        if len(insulin) < self.state_hist:
            insulin = np.concatenate((np.full(self.state_hist - len(insulin), -1), insulin))
        return_arr = [bg, insulin]
        return np.stack(return_arr).flatten()


    @property
    def action_space(self):
        ub = self.env.pump._params['max_basal']
        return spaces.Box(low=0, high=ub, shape=(1,))

#    @property
#    def observation_space(self):
#        return spaces.Box(low=0, high=np.inf, shape=(1,))

    @property
    def observation_space(self):
        st = self.get_state()
#        num_channels = int(len(st)/self.state_hist)
#        return spaces.Box(low=0, high=np.inf, shape=(num_channels, self.state_hist))
        return spaces.Box(low=0, high=np.inf, shape=(len(st),))
