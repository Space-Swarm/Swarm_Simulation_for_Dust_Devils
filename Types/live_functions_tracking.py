#imports
import sympy as sym
import plotly as py
import plotly.tools as tls
from plotly import graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from IPython.display import Image

import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html

import numpy as np
import pandas as pd
import ipywidgets as widgets
import time
import math

from scipy import special
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from collections import deque
from scipy.spatial.distance import cdist, pdist, euclidean
from sympy import sin, cos, symbols, lambdify

from robot import Robot
from dust_devil import DustDevil

import random
import datetime
import os
import math
import sys

import Processing_Functions_BP


import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html

import json
import pickle

from ast import literal_eval


#Function to initialise robots
def initialise(robot_number,length,initial,robot_speed,R,random_seed,lattice_constants):
    '''
    Returns the list of robot objects

            Parameters:
                    robot_number (int): Number of robots
                    length (int): Length of one side of the area

            Returns:
                   robots (list): List of robot objects
    '''
    robots = []
    X,Y = random_position(robot_number,-initial,initial,random_seed)
    cnt = 0
    for x,y in zip(X,Y):
        walk_time = int(np.random.normal(5,3,1))
        heading = random.uniform(0,2*math.pi)
        if(cnt == 0):
            identifier = 0
        else:
            identifier = 1
            cnt = -1
        robots.append(Robot(x,y,1.0,0,0,robot_speed, walk_time,heading,identifier,R,lattice_constants))
        print(identifier)
        cnt = cnt+1
    return robots

#Function to return randomised x,y position array
def random_position(robot_number,start,finish,random_seed):
    '''
    Returns random position array of x and y

            Parameters:
                    robot_number (int): Number of robots
                    start (int): Length of one side of the area

            Returns:
                   x (integer list): List of random x positions
                   y (integer list): List of random y positions
    '''
    np.random.seed(random_seed)
    x = list(np.random.normal(loc = 0, scale = finish, size = (robot_number)))
    y = list(np.random.normal(loc = 0, scale = finish, size = (robot_number)))
    """x = []
    y = []
    #generating random robot positions and appending them to a list which is returned
    for i in range(int(robot_number)):
        np.random.normal(loc = 0, scale = 20, size = 
        x.append(random.randint(start,finish))
        y.append(random.randint(start,finish))"""
    print("Initialising..")
    print(x)
    print(y)
    return x,y

#Function to return square grid of x,y positions
def grid_center(length,parts):
    '''
    Returns square grid position array of x and y

            Parameters:
                    length (int): Length of one side of the area
                    parts (int): how many grid points each length will be divided into

            Returns:
                   x (integer list): List of x positions in the grid
                   y (integer list): List of y positions in the grid
    '''
    #area being looked at for grid centre
    division = length/parts
    center_x = []
    center_y = []
    
    #two for loops that generate respective x and y grid co-ordinates
    for i in range(1,parts):
        for j in range(1,parts):
            center_x.append(division*j)
            center_y.append(division*i)
    return center_x,center_y

#Function to retrieve class object positions
def positions(objects):
    '''
    Returns 2D positions of class objects

            Parameters:
                    objects (list): List of objects

            Returns:
                   x (numpy array int): List of x positions of the object
                   y (numpy array int): List of y positions of the object
    '''
    x = []
    y = []
    for i in range(len(objects)):
       x.append(objects[i].x)
       y.append(objects[i].y) 
    return x,y

#Function to retrieve class object detection booleans
def detection(objects):
    '''
    Returns boolean list of detections

            Parameters:
                    objects (list): List of objects

            Returns:
                   detection (numpy array int): List of detection boolean values
    '''
    detected = []
    for i in objects:
        detected.append(i.detected)
    return detection


def broadcast(swarm, detection_list, set_R, multiply, R):
    #retrieving x and y positions
    x,y = positions(swarm)
   
    detecting = np.where(detection_list)
    list_detected = list(detecting[0])
    for i in list_detected:
        robot = swarm[i]
        x_copy,y_copy = x.copy(),y.copy()
        #finding the distance between the current robot and the other robots
        x_dist_unsorted = np.array(x_copy)-robot.x
        y_dist_unsorted = np.array(y_copy)-robot.y
        
        #calculating the distance from the robots to the current robot
        distance = np.sqrt(np.square(x_dist_unsorted)+np.square(y_dist_unsorted))
       
        #determining the robots within 1.5R distance
        distance_local = np.logical_and(distance>0,distance<multiply*R)   
        
        #creating the x and y distance matrices for robots within the local 1.5R distance
        x_dist = x_dist_unsorted[distance_local]
        y_dist = y_dist_unsorted[distance_local]
        
        neighbourhood = np.where(distance_local)
        list_neighbours = list(neighbourhood[0])

        neighbours = [swarm[i] for i in list_neighbours]
        #looping through the current robots neighbourhood
        for j in list_neighbours:
            swarm[j].R = set_R
    

#Function to check if the robot has gone over the edge and changing the value to negative if it has
def bounce(x_updated,y_updated,x_change,y_change,length):
    '''
    Returns bounced x and y position

            Parameters:
                    x_updated (float): X position after movement update
                    y_updated (float): Y position after movement update
                    x_change (float):  Change from previous X position
                    y_change (float):  Change from previous Y position
                    length (float):    Length of one side of the area

            Returns:
                   x_updated (float): Bounced X position
                   y_updated (float): Bounced Y position
    '''
    #if the x is larger than the side limit, then bounce off
    if(x_updated > length or x_updated < 0):
        x_updated = x_updated-2*x_change
    #if the y is larger than the side limit, then bounce off
    if(y_updated > length or y_updated < 0):
        y_updated = y_updated-2*y_change
    return x_updated,y_updated

#Function to work out the magnitude of a 2D vector
def magnitude(x,y):
    '''
    Finds the magnitude of 2 floats
    
            Parameters:
                   x (float): a float value
                   y (float): a float value
            
            Returns:
                   mag (np array): the resulting np array of magnitudes from the two given values

    '''
    mag = np.sqrt(np.square(x)+np.square(y))
    return mag

#Function to work out a 2D unit vector
def unit(vector):
    '''
    Finds the unit vector of a passed vector
    
            Parameters:
                   Vector(2D numpy array): the 2D position vector
            
            Returns:
                   Unit(2D numpy array): the resulting unit vector from the given vector

    '''
    #calculates the magnitude of the vector
    mag = magnitude(vector[0],vector[1])

    #using two if else statements to account for the possibility of values being zero, and then calculating the corresponding unit vector
    if(vector[0] == 0):
        x = 0
    else:
        x = vector[0]/mag
    if(vector[1]==0):
        y = 0
    else:
        y = vector[1]/mag
    unit = np.array([x,y])


    
    return unit

#Function to check if division by denominator is non zero, if it is zero, then the component is returned as zero
def division_check(numerator, denominator):
    '''
    Checks if the division is possible without zero division, if not set the component as zero
    
            Parameters:
                   Numerator (float): the numerator of the division
                 Denominator (float): the denominator of the division
            
            Returns:
                     Result (float): the result of the division

    '''

    if(denominator == 0):
        result = 0
    else:
        result = numerator/denominator
    
    return result

#Function to call physics based area coverage algorithm
def physics_walk(swarm,G,power,R,max_force,multiply,timestep):
    '''
    Updates the robots velocities based on a physicomimetics artificial gravity algorithm

            Parameters:
                   Swarm (list): a list of robot objects
                   G (float): the chosen G constant float value
                   power (float): the chosen power for the artificial gravity equation
                   R (float): the chosen R seperation float value
                   max_force (float): the maximum float limit on the force
                   multiply (float): the range multiplication float
            Returns:
                   None

    '''
    #setting initial constants
    force_x = 0
    force_y = 0 
    velocity_x = 0
    velocity_y = 0
    
    #retrieving x and y positions
    x,y = positions(swarm)
    
    #looping through swarm with each individual robot
    for i in range(len(swarm)):
        
        #selecting robots one by one
        robot = swarm[i]
        x_copy,y_copy = x.copy(),y.copy()
        
        #ensuring the own distance isn't included
        x_copy.pop(i)
        y_copy.pop(i)
        
        #initialising the forces as 0
        force = np.array([0,0])

        #finding the distance between the current robot and the other robots
        x_dist_unsorted = np.array(x_copy)-robot.x
        y_dist_unsorted = np.array(y_copy)-robot.y
        
        #calculating the distance from the robots to the current robot
        distance = np.sqrt(np.square(x_dist_unsorted)+np.square(y_dist_unsorted))

        #determining the robots within 1.5R distance
        distance_local = np.nonzero(distance) and (distance<multiply*R)
        

        #creating the x and y distance matrices for robots within the local 1.5R distance
        x_dist = x_dist_unsorted[distance_local]
        y_dist = y_dist_unsorted[distance_local]
        position = np.array([x_dist,y_dist])
        
        #initialising force changes
        force_change_dir = 0
        
        #number of robots within the neighbourhood
        robot_number = len(distance_local)
        

        #looping through the current robots neighbourhood
        for j in range(len(x_dist)):

            #storing current position 
            current_position = np.array([position[0,j],position[1,j]])


            #calculating magnitude of positions
            mag = magnitude(current_position[0],current_position[1])


            #calculating the force
            numerator = (G*(robot.mass**2))
            denominator = mag**2


            #calling function to account for division by zero, if there is a zero denominator, then the force component with zero is returned as zero
            force_change = division_check(numerator,denominator)

            #calculating the unit vector of the position
            distance_unit = unit(current_position)

            #if the magnitude is bigger than R, then a force is added to draw the robots together             
            if(mag>R):
                force_change_dir = distance_unit

            #if the magnitude is smaller than R, then a force is added to push the robots apart 
            elif(mag<R):
                force_change_dir = -distance_unit
            else:
                force_change_dir = np.array([0,0])

            #calculating new force change based on direction and magnitude
            force_delta = force_change_dir*force_change

            #constraining force to the maximum
            if((force_delta[0]**2+force_delta[1]**2)>((max_force)**2)):
                #calculating unit vector of the force
                unit_force = unit([force_delta[0],force_delta[1]])
                
                #multiplying it by the maximum force
                updating_force = unit_force*max_force
                
                #setting the new force equal to the respective updated maximum force components
                force_delta[0] = updating_force[0]
                force_delta[1] = updating_force[1]

            #calculating new force
            force = force+force_delta
            

        #calculating the change in velocity
        delta_vx = (force[0])*timestep/robot.mass
        delta_vy = (force[1])*timestep/robot.mass

        #calculating the new velocity
        velocity_x = robot.x_velocity + delta_vx
        velocity_y = robot.y_velocity + delta_vy

       #keeping the velocity within the maximum velocity of the robot
        if((velocity_x**2+velocity_y**2)>((robot.max_velocity)**2)):
            #calculating unit vector of the velocity
            unit_velocity = unit([velocity_x,velocity_y])
            
            #multiplying it by the maximum velocity
            velocity = unit_velocity*robot.max_velocity
            
            #setting the new velocity equal to the respective updated maximum velocity components
            velocity_x = velocity[0]
            velocity_y = velocity[1]

        #updating the robots velocity
        robot.update_velocity(velocity_x,velocity_y)    

#Function to return current dust devil performance metrics
def dust_check(dust_devils,robot,detection_range,timestep,set_R,swarm,multiply,R,countdown):    
    '''
    Checks if any of the robots detects a dust devil within range and increments the resulting performance metrics if so
            Parameters:
                   dust_devils (list): a list of dust devil objects
                   robot (object): the object of an individual robot
                   power (float): the chosen power for the artificial gravity equation
                   detecion_range (float): the detection range of the robots
            Returns:
                   collision_metric (float): the number of times a robot is within a dust devil
                   detection_metric (float): the number of new dust devils detected

    '''
    #the two performance metrics being returned
    collision_metric = 0
    detection_metric = 0
    
    #retrieving positions of the dust devils
    x_dust,y_dust = positions(dust_devils)
    
    #checking if dust devils have been generated yet
    if(len(x_dust)>0):
                #finding distance between the robot and the dust devils 
                x_dust_dist = np.array(x_dust.copy()) - robot.x
                y_dust_dist = np.array(y_dust.copy()) - robot.y
                
                #looking at the distances from the single robot to the dust devils
                distance_dust = np.sqrt(np.square(x_dust_dist)+np.square(y_dust_dist))

                #checking which dust devils are within the robots detection range
                boolean_detected = distance_dust<=detection_range
                indices_detected = np.nonzero(boolean_detected)

                #checking if the array of the dust devils detected is above zero
                if(indices_detected[0].size>0):
                    #looping through each index of dust devils detected
                    for index in (indices_detected):
                        
                        #incrementing the collision metric, which corresponds to 1 more detection of a dust devil
                        collision_metric = collision_metric + 1*timestep

                        #if the dust devils detected has not been detected before, increment the detection metric
                        if(dust_devils[index[0]].detected == False):
                            detection_metric = detection_metric + 1

                        #by default setting the detection to true, so it is not repeated in future detection metrics
                        dust_devils[index[0]].detected = True
                    robot.R = set_R
                    robot.mass = 1000
                    robot.detected = True
                    robot.countdown = countdown
                    detection_list = detection(swarm)
                    broadcast(swarm, detection_list, set_R,multiply,R)
                else:
                    robot.detected = False

    #returning the two metrics
    return collision_metric,detection_metric

def update_decay(swarm,R):
    #looping through all of the robots in the swarm
    for i in range(len(swarm)):
        if(swarm[i].countdown>0):
            swarm[i].countdown = swarm[i].countdown-1
            #print(swarm[i].countdown)
        if(swarm[i].countdown == 0 and swarm[i].R != R):
            swarm[i].detected = False
            swarm[i].R = R
            #print("Sucessfully changed")
            #print("R:", swarm[i].R)
            
#Function to update robot positions and metrics per time step
def update_timestep(swarm,dust_devils,timestep,frequency,min_neighbours,cluster_average,detection_range,R,multiply,set_R,countdown):
    '''
    Updating the dynamics of the simulation (robot positions, dust devil detections etc) per timestep
            Parameters:
                   swarm (list): a list of robot objects
                   dust_devils (list): a list of dust devil objects
                   timestep (float): the chosen timestep float (0.5s for example)
                   min_neighbours (list): the list of the average of the minimum neighbour distances
                   cluster_average (list): the list of the average of the cluster sizes
                   detection_range (float): the range which a robot can detect a dust devil
                   R (float): the virtual communication range of the robots
                   multiply (float): the range multiplication float
            Returns:
                   collision_metric (float): the number of times a robot is within a dust devil
                   detection_metric (float): the number of new dust devils detected

    '''
    #setting up the requirements for the performance metric calculations
    x,y = positions(swarm)
    collision_metric = 0
    detection_metric = 0
    minimum_distance = 0
    cluster_total = 0
    
    #looping through all of the robots in the swarm
    for i in range(len(swarm)):
        
        #selecting robots in the swarm
        robot = swarm[i]
        
 
        #setting up the cluster average and the minimum neighbour average metric
        x_copy,y_copy = x.copy(),y.copy()
        
        #removing the robots own position from the array
        x_copy.pop(i)
        y_copy.pop(i)
    
        #finding the distance between the current robot and the other robots
        x_dist_unsorted = np.array(x_copy)-robot.x
        y_dist_unsorted = np.array(y_copy)-robot.y
        
        #calculating the distance from the robots to the current robot
        distance = np.sqrt(np.square(x_dist_unsorted)+np.square(y_dist_unsorted))

        #calculating the minimum stance
        minimum_distance = minimum_distance + min(distance)
        
        #determining the robots within 1.5R distance
        distance_local = np.nonzero(distance) and (distance<multiply*R)
        
        #creating the x and y distance matrices for robots within the local 1.5R distance
        x_dist = x_dist_unsorted[distance_local]
        y_dist = y_dist_unsorted[distance_local]
        position = np.array([x_dist,y_dist])
        
        #calculating the cluster total
        cluster_total = cluster_total + cluster_function(x_dist,y_dist,R)
        
        #checking the dust devil detection method
        collision_change,detection_change = dust_check(dust_devils,robot,detection_range,timestep,set_R,swarm,multiply,R,countdown)
        collision_metric = collision_metric+collision_change
        detection_metric = detection_metric+detection_change  

        #calculating the change in position for x and y
        x_change = robot.x_velocity/frequency
        y_change = robot.y_velocity/frequency

        #calculating the new position
        x_updated = robot.x+x_change
        y_updated = robot.y+y_change

        #updating the position
        robot.update_position(x_updated,y_updated)
        
        
        
        
    #caculating the average of the minimum neighbour distance and appending it to a list
    avg_min_distance = minimum_distance/len(swarm)
    min_neighbours.append(avg_min_distance)
    
    #calculating the average of the clusters and appending it to a list
    cluster_avg = cluster_total/len(swarm)
    cluster_average.append(cluster_avg)    
   
    return collision_metric,detection_metric
            
#calculates the cluster average
def cluster_function(x_dist,y_dist,R):
    '''
    Calculating the size of the clusters within the neighbourhood of one robot
            Parameters:
                   x_dist (list): list of the x positions of the neighbouring robots
                   y_dist (list): list of the y positions of the neighbouring robots
            Returns:
                   cluster_count (float): the size of the cluster within the neighbourhood of the given robot

    '''
    
    #creating numpy array of positions for easier data manipulation
    distance = np.array([x_dist,y_dist])
    
    #setting the default cluster size as 1
    cluster_count = 1
    
    #looping through the distances of the robots within the local neighbourhood
    for j in range(len(x_dist)):
        #storing current distance from robot 
        current_distance = np.array([distance[0,j],distance[1,j]])

        #calculating magnitude of distance from the robot
        mag = magnitude(current_distance[0],current_distance[1])

        #checking if in the same cluster, if so adding to cluster count
        if(mag<0.2*R and mag!=0):
            cluster_count = cluster_count+1

    #returning the size of the cluster
    return cluster_count


#Function to calculate the G transition parameter
def G_transition(max_force,R,power):
    '''
    Calculates the transition value between liquid and solid states for the G parameter

            Parameters:
                   max_force (float): the maximum float limit on the force
                   R (float): the chosen R seperation float value
                   power (float): the chosen power for the artificial gravity equation
                   
            Returns:
                   G_transition (float): the G transition value for the current swarm setup

    '''
    G_transition = (max_force*(R**power))/(2*math.sqrt(3))
    return G_transition


#Function to calculate distance between a robot and the swarm
def dist(robot, x,y):
    '''
    Calculates the distances between a robot and the swarm

            Parameters:
                   Robot (object): the robot object that this calculation is for
                   x (float): array of robot x positions
                   y (float): array of robot y positions
                   
            Returns:
                   x_dist (list): an array of the x distances from the current robot to the robot in the swarm
                   y_dist (list): an array of the y distances from the current robot to the robot in the swarm

    '''
    #finding the distance between the current robot and the other robots
    x_dist = x-robot.x
    y_dist = y-robot.y
    return x_dist,y_dist


#Function to create random walk
def random_walk(swarm):
        """
        Updates the movement of the robot by checking if the current heading needs to be reset, and then updating the position           according to the header

            Parameters:
                   Swarm (list): a list of robot objects
            
            Returns:
                   None
        """
        weighting = 20
        deviation = 5
    
        for robot in swarm:
            if robot.check_walk():
                walk_time = int(np.random.normal(weighting,deviation,1))
                heading = random.uniform(0,2*math.pi)
                robot.walk_reset(walk_time,heading)
            x_change = math.cos(robot.heading)*robot.x_velocity
            y_change = math.sin(robot.heading)*robot.y_velocity
            x_updated = robot.x + x_change
            y_updated = robot.y + y_change
            #self.bounce(x_updated,y_updated,x_change,y_change,60000)
            robot.update_position(x_updated,y_updated)
            robot.update_timer()



#Function to update/create dust devils
def dust(dust_devils,probability_dust,side,timer,dust_speed,dust_time,timestep,frequency):
    '''
    Adds randomly generated dust devil object to list according to a given probability, pops the dust devils when their time is up and returns the count of the number of new dust devils 

            Parameters:
                   dust_devils (list): a list of dust devil objects
                   probability_dust (float): the probability that a dust devil will appear within the constrained area
                   side (float): the length in km of one side of the area being examined
                   timer (float): the current timestamp of the simulation
                   dust_speed (float): the set speed of the dust devils
                   dust_time (integer): an integer of the time a dust devil lasts
                   timestep (float): the timestep that is currently being used
                   frequency (float): 1/timestep, which is used for length of dust devil time in simulation for instance
            Returns:
                   count (integer): a count of the number of dust devils generated


    '''
    
    #initialising count, which is set 0 as default
    count = 0
    
    #generating a random float between 0 and 1
    generated = random.uniform(0,1)
    amount = len(dust_devils)
    
    if(timer == 0):
    #checking if a dust devil should be generated according to the random float generated
    #if(generated<probability_dust):
        
        
        #generating a random number within the maximum domain of x and y
        x = random.randint(-side*1000,side*1000)
        y = random.randint(-side*1000,side*1000)
        
        #generating a dust devil radius
        radius = random.randint(0,500)
        x_trajectory,y_trajectory = trajectory_dust(dust_speed,timestep)
        
        #adding to the dust devils list
        dust_devils.append(DustDevil(x,y,radius,dust_speed,timer,dust_time*frequency,x_trajectory,y_trajectory))
        
        #setting the count as 1, since 1 dust devil has been created
        count = 1
        

    #initialising i for use in a while loop
    i = 0
    
    #using i as the termination paramter for the while loop as a for loop condition can't be changed within the loop when the dust devil length changes
    while i<len(dust_devils):
        
        #checking if there is a dust devil has reached the end of its life, and removing it from the list if so
        if(dust_devils[i].end_time < timer ):
            dust_devils.pop(i)
            #print("Dust Devil Popper: ", [dust_devils[i].x,dust_devils[i].y])
            #decrementing the i value used in the while loop, as this prevents the loop from skipping over a dust devils once one is removed
            i-=1
        #every iteration increases the i value until termination condition is hit
        i+=1
    
    #returning number of dust devils generated, which is either a 0 or a 1
    return count

#generating the trajectory of the dust devil
def trajectory_dust(dust_speed,timestep):
    '''
    Generates the equations of motion for the dust devils
    
            Parameters:
                   dust_speed (integer): the dust devils speed
                   timestep(float): the virtual timestep the simulation is being updated by 
                   
            Returns:
                   x_trajectory (SymPy Expression): the parametric equation of motion for the x position of the dust devil
                   y_trajectory (SymPy Expression): the parametric equation of motion for the y position of the dust devil

    '''
    
    #representing the trajectory in a x,y and r coordinates system using symbols
    x = sym.Symbol('x')
    y = sym.Symbol('y')
    r = dust_speed*timestep
    
    #selecting which trajectory case will be generated
    case = 0
    
    #randomly generating the heading
    heading = random.uniform(0,2*math.pi)
    
    #only active case used
    if(case == 0):
        #representing x and y trajectory with parametric equation for straight line
        x_trajectory = x + r*math.cos(heading)
        y_trajectory = y + r*math.sin(heading)
        
    #placeholder case
    elif(case == 1):
        x_trajectory = x 
        y_trajectory = y
    #another placeholder case
    else:
        x_trajectory = x
        y_trajectory = y
        
    #returning the trajectory equations
    return x_trajectory,y_trajectory

#function to update dust devil positions based on trajectory 
def update_dust(dust_devils):
    '''
    Updates the positions of the dust devil based on the equation of motion
    
            Parameters:
                   dust_devils (list): list of the dust devil objects

                   
            Returns:
                   None
    '''
    #initialising symbolic math for use in the expressions
    x = sym.Symbol('x')
    y = sym.Symbol('y')
    
    #looping through every dust devil
    for dust in dust_devils:

        #loading the equations of motion
        x_trajectory = dust.x_trajectory
        y_trajectory = dust.y_trajectory
        
        #generating corresponding function expressions for the equations
        x_function = lambdify(x, x_trajectory, 'numpy')
        y_function = lambdify(y, y_trajectory, 'numpy')
        
        #evaluating the equations using the dust devil positions
        x_updated = x_function(dust.x)
        y_updated = y_function(dust.y)

        #updating the dust devil positions
        dust.update_position(x_updated,y_updated)


#Function to load initial positions
def load_positions(path,time):
    '''
    Loading the swarm positions based on a path and a timestep
    
            Parameters:
                   path (string): a string of the directory where the data is being loaded from
                   time (int): an integer corresponding to the desired timestamp for the loaded data
            Returns:
                   x (numpy array): a numpy array of the loaded x positions
                   y (numpy array): a numpy array of the loaded y positions
    '''
    #loading the data
    positions = np.load(path)
    
    #accessing the specific position data at the timestamp
    final = positions[:,:,time]
    
    #breaking the data down into the x and y components
    x = final[0]
    y = final[1]
    
    #returning the x and y components
    return x,y

#Function to initialise robots based on final positions
def pre_initialise(X,Y,robot_speed,R):
    '''
    Loading the swarm positions based on a path and a timestep
    
            Parameters:
                   X (numpy array): a numpy array of the preloaded x positions
                   Y (numpy array): a numpy array of the preloaded y positions
                   robot_speed (integer): a integer of the robot speed
            Returns:
                   swarm (list): a list of the robot objects with the corresponding loaded start positions
    '''
    swarm = []
    
    #count is used for the identifier of the robot
    cnt = 0
    
    #looping through the positions to initialise the robot swarm
    for x,y in zip(X,Y):
        
        #initialising the random walk time and heading parameters for the robot
        walk_time = int(np.random.normal(5,3,1))
        heading = random.uniform(0,2*math.pi)
        
        #splitting the swarm into half of one specific type, and half of another
        if(cnt == 0):
            identifier = 0
        else:
            identifier = 1
            cnt = -1
            
        #appending the robot objects
        swarm.append(Robot(x,y,1,0,0,robot_speed, walk_time,heading,identifier,R))
        
        #incrementing count, used to ensure there is an equal split of identifies in robots
        cnt = cnt+1
        
    #returning the list of the robot objects
    return swarm
