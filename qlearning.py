import numpy as np 
import torch 
from collections import defaultdict

class Envman:
  def __init__(self, size , psize):
    self.psize = psize
    self.lambdavalue = 0.01
    self.n = size 
    self.probability = np.zeros((4, size, size))
    self.probability  = self.generate_qfunction(self.probability)
    self.action_state_value = np.zeros((4 , self.n))
    
  def move_state(self, state, action):
    move_probability=self.probability[action,state , :]
    cumulative_sum = np.cumsum(move_probability)
    cumulative_sum = np.ceil(cumulative_sum*1000).astype(int)
    total_range = [] 
    next_state = None 
    for ii in range(0, self.n):
      if ii == 0 :
        total_range.append([0,  cumulative_sum[0].item()])
      else:
        total_range.append([ cumulative_sum[ii-1].item(),cumulative_sum[ii].item() ])
    
  
    pick=np.random.choice(range(1001), size=1 , replace=False)
    for idx , jj in enumerate(total_range):
      if (pick > jj[0]) and (pick <= jj[1]):
        next_state = idx 
        break 
      else:
        pass
    return next_state
    
    
  def select_state_action_value(self, state, action ):
    return  self.action_state_value[action, state] 
  
  def update_state_action_value(self, state, action, value):
    self.action_state_value[action, state] =  (1-self.lambdavalue)*self.action_state_value[action, state] + self.lambdavalue*value
    
  def generate_qfunction(self, tmp):
    for j in range(0,4):
      for i in range(0,self.n):
        unique_integers = np.random.choice(range(self.psize+1), size=self.n , replace=False)
        cal_probabiliy = unique_integers / sum(unique_integers)
        tmp[j, i, : ] = cal_probabiliy
    return tmp
  
    


if __name__ == "__main__":
  envagent=Envman(6,40)
  envagent.move_state(0, 1)
