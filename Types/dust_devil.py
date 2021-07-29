#!/usr/bin/env python
# coding: utf-8

# In[2]:


import time


class DustDevil:
    """
         A class to represent a dust devil

         ...

         Attributes
         ----------
           x : float
              x position
           y : float
              y position
      radius : int
              dust devil radius
        speed: float
              speed of dust devil
   start : float
              timestep of first initialisation of dust devil
end_time : float
              time that the dust devil lasts until
    

         Methods
         -------
         update_position(x_position,y_position):
             Updates the x and y positions of the dust devil
             
        
     """
    def __init__(self, x_position, y_position, radius, speed,timestep,survival_time,x_trajectory,y_trajectory):
        """
        Initialises the dust devil object and its attributes

        Parameters
        ----------
        x : float
          x position
        y : float
          y position
   radius : int
          dust devil radius
     speed: float
           speed of dust devil
     start : float
           timestep of first initialisation of dust devil
end_time : float
            time that the dust devil lasts until
x_trajectory : sym math
            x trajectory
y_trajectory : sym math
            y trajectory
         """
        self.x = x_position
        self.y = y_position
        self.radius = radius
        self.speed = speed
        self.start = timestep
        self.end_time = timestep+survival_time
        self.x_trajectory = x_trajectory
        self.y_trajectory = y_trajectory
        self.detected = False
        
    def update_position(self,x_position,y_position):
        """
        Updates the x and y positions of the dust devil

        Parameters
        ----------
            x_position : float
                x position of the dust devil
            y_position : float
                y position of the dust devil
                
        Returns
        -------
            None
        """
        self.x = x_position
        self.y = y_position
    


# In[ ]:




