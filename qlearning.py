import numpy as np 
import torch 
from collections import defaultdict
import random

class Envman:
  def __init__(self, size , psize , board):
    self.psize = psize
    
    #self total direction 
    self.direction = [0,1,2,3]
    
    # up, down, left, right
    self.actions = [[-1,0], [1,0], [0,-1], [0,1]]
    
    #start and end 
    self.end_position = [[0,0], [size-1, 0]] 
    
    #current position
    self.current_state = [0,0]

    #update learning value 
    self.lambdavalue = 0.01
    self.n = size 
    
    #board
    self.board = board
    
    #hole
    self.holes = self.find_hole()
    
    #start position 
    self.total_state = [[i,j] for i in range(0, size) for j in range(0,size) if [i, j] not in (self.holes+[[0, 0]]+ [[6,0]]) ]
    
    #probability
    self.probability = np.zeros((size, size, 4, 4))
    
    
    #mdp generation 
    self.probability  = self.generate_mdpfunction(self.probability)
    
    #q_value
    self.action_state_value = np.zeros((4 , self.n))
    
    
  
  def find_hole(self):
    tmp = []
    for i in range(0, self.n):
      for j in range(0,self.n):
        if self.board[i, j] < 0:
            tmp.append([i,j])
    return tmp
  
  
  def initialization(self):
     self.current_state  = random.choice(self.total_state)
     

  def availibility(self,  current ):
    possible = [] 
    for idx, move in enumerate(self.actions):
      next_x = current[0]+move[0]
      next_y = current[1]+move[1]
      if (next_x < self.n) and (next_x >= 0 ) and (next_y< self.n) and(next_y >=0):
        possible.append(idx)  
    possible.sort()    
    return possible
      
  def step(self):
    pass
    
  # def move_state(self, state, action):
  #   move_probability=self.probability[action,state , :]
  #   cumulative_sum = np.cumsum(move_probability)
  #   cumulative_sum = np.ceil(cumulative_sum*1000).astype(int)
  #   total_range = [] 
  #   next_state = None 
  #   for ii in range(0, self.n):
  #     if ii == 0 :
  #       total_range.append([0,  cumulative_sum[0].item()])
  #     else:
  #       total_range.append([ cumulative_sum[ii-1].item(),cumulative_sum[ii].item() ])
    
  
  #   pick=np.random.choice(range(1001), size=1 , replace=False)
  #   for idx , jj in enumerate(total_range):
  #     if (pick > jj[0]) and (pick <= jj[1]):
  #       next_state = idx 
  #       break 
  #     else:
  #       pass
  #   return next_state
    
    
  def select_state_action_value(self, state, action ):
    return  self.action_state_value[action, state] 
  
  def update_state_action_value(self, state, action, value):
    self.action_state_value[action, state] =  (1-self.lambdavalue)*self.action_state_value[action, state] + self.lambdavalue*value
    
  def generate_mdpfunction(self, tmp):
    self.beta = 0.37
    for i in range(0,self.n):
      for j in range(0,self.n):
        for k in range(0, 4):
          move_possible=self.availibility([i,j])
          unique_integers = np.random.choice(range(self.psize+1), size=len(move_possible), replace=False)
          cal_probabiliy = (unique_integers / sum(unique_integers))*self.beta
          add_to_direction=1-sum(cal_probabiliy)
          for q , pro in zip(move_possible, cal_probabiliy):
            self.probability[i,j,k, q] =  pro
          self.probability[i,j,k,k] += add_to_direction
    return tmp
  



def hole(board):
  board[board.shape[0]-1 , 0 ] = 20
  for i in range(2,3):
    board[1 , i ] = -40
  
  for i in range(1,2):
    board[5 , i ] = -40
  
    
  for i in range(6,7):
    board[0 , i ] = -40
    
  for i in range(6,7):
    board[1 , i ] = -40
  
  for i in range(1,4):
    board[3 , i ] = -40
  
  for i in range(1,5):
    board[4 , i ] = -40
    
  for i in range(1,2):
    board[5 , i ] = -40
  
  
  for i in range(1,2):
    board[2 , 1 ] = -40
  
  return board

if __name__ == "__main__":
  board = np.zeros((7,7))
  board = hole(board)
  envagent=Envman(7, 40, board)
