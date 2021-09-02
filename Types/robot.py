#!/usr/bin/env python
# coding: utf-8

# In[3]:
import numpy as np
import time
import random
import math

class Robot:
    """
         A class to represent a robot

         ...

          Attributes
          ----------
         x : float
           x position
         y : float
           y position
         mass : float
           mass of the robot in Kg
x_velocity : float
           velocity of robot in the x direction
y_velocity : float
           velocity of robot in the y direction
max_velocity : float
           max velocity of the robots
 walk_time : float
           time that the robot will walk in a certain direction for  
   heading : float
           angle that the robot is heading   
   weighting: float
            Gaussian weight for time to walk
       identifier: int
            the type of particle, for lattice structure purposes
    

          Methods
          -------
          update_position(x_position,y_position):
              Updates the x and y positions of the robot
             
          update_timer():
              Updates the timer by 1
             
          walk_reset():
              Resetting the timer, signalling the start of a new direction
             
          check_walk():
              Checks if the walk has been completed for the given walk_time
             
          update_movement():
              Implements a timestep change in movement for the robots
             
          control():
              Implementing movement of robot and updating the timer
             
           bounce(x_updated,y_updated,x_change,y_change,length):
               Bounces the robot off the edge if it crosses the outer limits
              
        
    """
    def __init__(self,x_position,y_position,mass,x_velocity,y_velocity,max_velocity,walk_time,heading,identifier,R,lattice_constants,G):
        """
        Initialises the robot object and its attributes

        Parameters
        ----------
         x : float
           x position
         y : float
           y position
         mass : float
           mass of the robot in Kg
x_velocity : float
           velocity of robot in the x direction
y_velocity : float
           velocity of robot in the y direction
max_velocity : float
           max velocity of the robots
 walk_time : float
           time that the robot will walk in a certain direction for  
   heading : float
           angle that the robot is heading   
   weighting: float
            Gaussian weight for time to walk
  identifier: int
            the type of particle, for lattice structure purposes
          R : float
            the comparison boundary which creates attractive/repulsive forces
       detected : boolean
            a boolean that is true if a dust devil is being detected
        
         """
        self.x = x_position
        self.y = y_position
        self.mass = mass
        self.x_velocity = x_velocity
        self.y_velocity = y_velocity
        self.max_velocity = max_velocity
        self.walk_time = walk_time
        self.heading = heading
        self.timer = 0
        self.weighting = 100
        self.identifier = identifier
        self.R = R
        self.detected = False
        self.countdown = 0
        self.counter = 0
        self.lattice_constants = lattice_constants
        self.lattice = lattice_constants[0]
        self.honeycomb = True
        self.G = G
        
    def update_position(self,x_position,y_position):
        """
        Updates the x and y positions of the robot

        Parameters
        ----------
          x_position : float
              x position of the robot
          y_position : float
              y position of the robot
              
        Returns
        -------
            None
        """
        self.x = x_position
        self.y = y_position
   
           
        
    def update_velocity(self,x_velocity,y_velocity):
        """
        Updates the x and y velocity components of the robot

        Parameters
        ----------
          x_velocity : float
              x velocity component of the robot
          y_velocity : float
              y velocity component of the robot
              
        Returns
        -------
            None
        """
        self.x_velocity = x_velocity
        self.y_velocity = y_velocity
        
    def update_timer(self):
        """
        Updates the timer by incrementing it by 1

        Parameters
        ----------
            None
          
        Returns
        -------
            None
        """
        self.timer = self.timer+1
    
    def update_counter(self):
        """
        Updates the counter by incrementing it by 1

        Parameters
        ----------
            None
          
        Returns
        -------
            None
        """
        self.counter = self.counter+1
        """if(self.counter == 5 or self.counter == 6 or self.counter == 7 or self.counter == 4):
            self.honeycomb = False
            self.set_lattice_mode()
        elif(self.counter>10):
            self.reset_counter()
            self.set_lattice_mode()
        else:
            self.honeycomb = True
            self.set_lattice_mode()"""
        if(self.counter == 5 or self.counter == 6 or self.counter == 7 or self.counter == 4):
            self.honeycomb = False
            self.set_lattice_mode()
        elif(self.counter>10):
            self.reset_counter()
            self.set_lattice_mode()
        else:
            self.honeycomb = True
            self.set_lattice_mode()
    
    def set_lattice_mode(self):
        if(self.honeycomb):
            self.lattice = self.lattice_constants[0]
        else:
            self.lattice = self.lattice_constants[1]
    def set_lattice_switch(self):
        self.Honeycomb = not self.Honeycomb
        if(self.honeycomb):
            self.lattice = self.lattice_constants[0]
        else:
            self.lattice = self.lattice_constants[1]
    def reset_counter(self):
        """
        Resets the counter to 0

        Parameters
        ----------
            None
          
        Returns
        -------
            None
        """
        self.counter = -10
        
        
    def walk_reset(self,walk_time,heading):
        """
        Resets the walk_time, the heading and the timer

        Parameters
        ----------
            None
          
        Returns
        -------
            None
        """
        self.walk_time = walk_time 
        self.heading = heading 
        self.timer = 0
        
    def check_walk(self):
        """
        Checks if the time has elapsed past the threshold for the current robot heading

        Parameters
        ----------
            None
          
        Returns
        -------
            True or False boolean
        """
        #compares the walk time to the current timer for the robot, and resets if completed
        if(self.walk_time<self.timer):
            return True
        else:
            return False
            
    
    def bounce(self,x_updated,y_updated,x_change,y_change,length):
        """
        Returns bounced x and y position

        Parameters
        ----------
            x_updated (float): X position after movement update
            y_updated (float): Y position after movement update
            x_change (float):  Change from previous X position
            y_change (float):  Change from previous Y position
            length (float):    Length of one side of the area
          
        Returns
        -------
            x_updated (float): Bounced X position
            y_updated (float): Bounced Y position
        """
        #if the x is larger than the side limit, then bounce off
        if(x_updated > length or x_updated < 0):
            self.heading = -self.heading
            self.update_movement
        #if the y is larger than the side limit, then bounce off
        if(y_updated > length or y_updated < 0):
            self.heading = -self.heading
            self.update_movement

        
                  

# In[ ]:





# In[ ]:




