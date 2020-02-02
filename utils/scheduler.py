import math
from functools import partial

import numpy as np

from utils.other_utils import set_lr, set_momentum
        
        
class LR_Scheduler:
    def __init__(self, schedulers):
        self.schedulers = schedulers
        self.current_scheduler = self.schedulers[0]
        self.current_position = 0
        
    def step(self):
        if self.finished() != True:
            if self.current_scheduler.finished() == True:
                self.current_position += 1
                self.current_scheduler = self.schedulers[self.current_position]
            self.current_scheduler.step()
            
    def finished(self):
        return self.schedulers[-1].finished() == True    
    
    
class SchedulerConst:
    def __init__(self, optimizer, steps):  
        self.current_step = 0
        self.steps = steps
    
    def step(self):
        self.current_step += 1
            
    def finished(self):
        return self.current_step >= self.steps     
    
    
class SchedulerCosine:
    def __init__(self, optimizer, start_lr, finish_lr, lr_mult, steps, start_mom=None, finish_mom=None):  
        self.current_step = 0
        if (finish_mom  is not None) & (start_mom is not None):
            self.mom_coeff = (finish_mom - start_mom) / steps            
            if 'momentum' in optimizer.param_groups[0].keys():
                self.mom_name = 'momentum'
            elif 'betas' in optimizer.param_groups[0].keys():    
                self.mom_name = 'betas'
        self.lr_mult = lr_mult
        self.optimizer = optimizer
        self.scheduler = sched_cos(start_lr, finish_lr)
        self.start_mom = start_mom
        self.steps = steps
    
    def step(self):
        self.current_step += 1
        if self.current_step <= self.steps:
            set_lr(self.optimizer, np.round(self.lr_mult * self.scheduler(self.current_step/self.steps), 10)) 
            if hasattr(self, 'mom_coeff'):
                set_momentum(self.optimizer, self.mom_name, self.start_mom  + self.mom_coeff * self.current_step)
            
    def finished(self):
        return self.current_step >= self.steps  
    
    
def annealer(f):
    def _inner(start, end): return partial(f, start, end)
    return _inner

@annealer
def sched_cos(start, end, pos): return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2
    