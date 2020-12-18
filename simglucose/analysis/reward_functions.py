import numpy as np

def magni_reward(bg_hist, **kwargs):
    bg = max(1, bg_hist[-1])
    fBG = 3.5506*(np.log(bg)**.8353-3.7932)
    risk = 10 * (fBG)**2
    return -1*risk