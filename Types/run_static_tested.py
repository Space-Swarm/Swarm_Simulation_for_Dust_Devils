#!/usr/bin/env python
# coding: utf-8

# # Swarm Simulator for Outer Space
# 
# This swarm simulator will explore different swarm algorithms for an application in outer space. This approach will divide a plot into grid squares, which can then be analysed to determine and control the behaviour of each individual agent in the swarm.

# In[1]:


#imports

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
from live_functions_tracking import pre_initialise_types,update_decay, initialise,random_position, grid_center,positions,bounce,magnitude,unit,division_check,physics_walk,dust_check,update_timestep,cluster_function,G_transition,dist,random_walk,dust,trajectory_dust,update_dust,load_positions,pre_initialise

import random
import datetime
import os
import math
import sys

import Processing_Functions_Tracking


import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html

import json
import pickle

from ast import literal_eval

app = dash.Dash(__name__)


# In[2]:


print("Kernel Check")


# In[3]:


#Plotly offline mode
init_notebook_mode(connected=True)


# In[4]:


#function to calculate area coverage using grid metric
def grid_coverage(grid,positions,length):
    #creating the grid bins for area coverage
    x_bins = np.linspace(grid[0],grid[1],length+1)
    y_bins = np.linspace(grid[0],grid[1],length+1)
    
    x_positions = positions[0]
    y_positions = positions[1]
    #initialising empty coordinates array for use in the grid metric
    coordinates = []
    for i in range(len(x_positions)):
        #setting current position
        x_current = x_positions[i]
        y_current = y_positions[i]
        if(grid[0]<=x_current<=grid[1] and grid[0]<=y_current<=grid[1]):
            #searching for the x anf y coordinates using the numpy search sorted function combined with the previously defined bins
            x = np.searchsorted(x_bins, x_current, side ='left')
            y = np.searchsorted(y_bins, y_current, side ='left')
            coordinates.append((x,y))

            print(coordinates)


    rm_duplicates = list(set(coordinates)) #set is used to remove duplicates, as sets require distinct elements
    area_coverage = len(rm_duplicates) #converting it to a percentage
    total = length*length
    return area_coverage


# In[ ]:





# In[5]:


def check_distance(robot,R,distance_array, multiply,like):
    positions = []
    for displacement in distance_array:
        robot.update_counter()
        if(like):
            if(dispalcement > 0 and (displacement*(1/robot.lattice))<multiply*R):
                positions.append(displacement)
        else:
            if(displacement > 0 and displacement<multiply*R):
                positions.append(displacement)
    return np.array(positions)


# In[6]:


"""positions = [[379.46846005717373, 438.02847717862016, 153.89596436963546, -193.1283557781956, -416.6839770913143, 34.081562711073445, 374.72181409793734, 4.2575140859884275, 544.4866167789752, 23.190850898954615, 206.292374293048, 459.30724154559175, -466.8672644979862, -150.38320663356518, -397.21668016133697, -371.0739121077903, 478.0788958724223, 282.45802392448843, -384.6534189059488, -144.80793054458428, -106.02592213527876, -444.43440341272503, -170.22569193162536, 382.92004456397865, 62.87070951518658, 212.95973786111648, -278.0567838144116, 115.1011207349581, -130.45332167322687, 205.04799777830556, 365.2264611175408, -23.515984683000326, 445.35388090506376, 98.53544732837173, 290.1076391910268, 202.11715479254326, 78.77032842771135, -42.89735025816067, -213.6628351057763, -35.500698734810456, 267.70466918435454, -485.64700248216207, 361.8527778542226, -296.71363260442513, -338.56952980783785, 165.3445238355776, 263.37547521526204, 359.43633016009665, 467.6646740557361, 104.77624573139161, 22.802832028931817, 52.73052471062435, -90.0057929627286, -98.59782691890629, -227.5188903050415, -573.2341830580037, -580.8326772744379, -13.063102460866759, 253.48036728075232, 161.01711053412822, -204.2595823910022, -480.59403217739134, 80.88401070149266, -608.1992177828363, 297.83433951064205, -316.71542444098327, 131.36911256856305, -497.0272951287137, -290.5273579759531, -409.8659901666948, 47.456347223082176, 323.3539728944321, -543.7684205850073, 379.78709845410674, 121.72832157016853, -209.01836032758393, 535.1602882987758, 549.207657699739, -254.10778066703574, -273.7172482391562, 48.06370774024232, -330.92367051605993, -54.73831221701292, 291.9691912442395, -47.0547013703526, 288.4613827402791, -312.1418887461871, 182.28356368866713, -398.5758594157063, -67.27356634825496, 0.031665638870082, -205.99973061025287, -509.33761911861745, 278.0730396276819, 189.7830598804298, 451.72116568297275, 198.535741465554, -121.00226946769848, -40.98085357848208, 337.6674334402037], [193.02754782914445, 454.5042113341391, -100.92137910004897, 42.73268937369279, -49.679767542452396, -410.48798824719745, 81.82807907994734, 445.55917342798534, 89.96929697234395, 250.0259556170741, 87.41833872075051, 133.3384328017171, -229.45480008921845, -505.01137623725106, 118.75792908404702, -262.13214566288497, 250.2075485728992, 436.6719202797577, 352.7205989049511, 209.42180023460577, 21.884694777627473, 40.25440129090552, 303.0328217899561, 289.9058830186664, 529.0254004423562, 316.7822320939215, -277.6194169251578, 76.21550582522042, -278.7533095286261, -366.26268424563386, -215.05528710142192, 81.30701753336464, -369.010993564116, 366.92015463401185, -60.17686820154037, -14.09842721581898, 169.2458360853655, -231.16839481272285, -109.10706887520573, 563.5289509758492, 238.49224250257612, 336.99743083799547, 384.98556185196446, 129.87691498490432, -165.3777398733705, 523.7653545787055, 537.7224160878344, -311.88547773144955, -65.77239858220321, -250.85317375837485, -309.74202514256876, -88.89297412647363, 475.213208992796, 375.2858493228242, -439.3506909932829, 163.32487662995197, -102.10636450206604, -552.9445461095431, -554.1950272403276, -531.5171372566158, 138.26975695917804, -124.34040953094612, -592.5956897313303, -10.72837940030393, -392.0804717921708, -71.05680996082148, -433.6177162693273, -325.55224682922926, 27.009085123259204, 219.32524148135457, -183.00076288226694, -483.74204352537174, 68.96170014446278, -17.259738974716335, 274.068202881296, -340.99064542325107, -223.6484491607289, -12.409889257920756, 422.77390658817546, 323.63748322858896, 12.491227031598967, -424.87210080149634, -460.97296804859513, 144.38365899621817, 182.50733406961103, 38.0908085787018, 230.96593987619093, 186.02118863703964, -359.63510863792385, -358.44498807936657, 347.51582237356797, -206.7621265707475, 240.1251116438812, -163.5550254008532, 409.48407753219686, -168.65554545819106, -271.3737154444477, -77.09110655116845, -135.18071874470147, 597.2627315131041]]
x_0 = positions[0]
y_0 = positions[1]
print((grid_coverage(np.array([-500,500]),np.array([x_0,y_0]),10)))"""


# In[7]:



#Function to call physics based area coverage algorithm
def physics_walk_honey(swarm,power,max_force,multiply,timestep,lattice_constants,velocity_coefficient,max_force_overall):
    '''
    Updates the robot position based on a physicomimetics algorithm

            Parameters:
                   Swarm (list): a list of robot objects
            
            Returns:
                   None

    '''

    #print("Physics Called \n===============")
    #setting initial constants
    #setting initial constants
    force_x = 0
    force_y = 0 
    velocity_x = 0
    velocity_y = 0
    x,y = np.array(positions(swarm))
    type_0,type_1 = split(swarm)
    x_0,y_0 = np.array(positions(type_0))
    x_1,y_1 = np.array(positions(type_1))
    #print("Length 0:",len(x_0))
    #print("Length 1: ",len(x_1))
    
    #looping through each robot in the swarm
    for i in range(len(swarm)):
        force_0 = np.array([0,0])
        force_1 = np.array([0,0])
        #selecting robots one by one
        robot = swarm[i]
        R = robot.R
        robot.update_velocity(velocity_coefficient*velocity_x,velocity_coefficient*velocity_y)    
        #initialising the forces as 0
        force = np.array([0,0])
        #x_dist_unsorted,y_dist_unsorted = dist(robot,swarm)
        
        #finding the distance between the current robot and the other robots
        delta_x0,delta_y0 = dist(robot,x_0,y_0)
        #print([delta_x0,delta_y0])
        delta_x1,delta_y1 = dist(robot,x_1,y_1)
        """#finding the distance between the current robot and the other robots
        delta_x0,delta_y0 = dist(robot,x,y)
        delta_x1,delta_y1 = dist(robot,x,y)"""
        
        #calculating the distance from the robots to the current robot, split up according to robot type
        distance_0 = np.sqrt(np.square(delta_x0)+np.square(delta_y0))
        distance_1 = np.sqrt(np.square(delta_x1)+np.square(delta_y1))
        
        #finding the corresponding booleans
        multiply_like = 1.3
        multiply_unlike = 1.7
        
        robot.update_counter()
        constant = 1/robot.lattice
        scale = R/20
        if(robot.identifier == 0):
            distance_local_0 = np.logical_and(distance_0>0,(distance_0*constant)<multiply_like*R)
            distance_local_1 = np.logical_and(distance_1>0,distance_1<multiply_unlike*R)
            like_0 = True
            like_1 = False
            neighbourhood_0 = np.where(distance_local_0)
            #print("Raw Indices: ", neighbourhood_0)
            list_neighbours_0 = list(neighbourhood_0[0])
            neighbours_0 = [type_0[k] for k in list_neighbours_0]
            #print(list_neighbours_0)
            
            
            neighbourhood_1 = np.where(distance_local_1)
            #print("Raw Indices: ", neighbourhood_1)
            list_neighbours_1 = list(neighbourhood_1[0])
            neighbours_1 = [type_1[l] for l in list_neighbours_1]
            #print(list_neighbours_1)
            
        if(robot.identifier == 1):
            like_0 = False
            like_1 = True
            distance_local_0 = np.logical_and(distance_0>0,distance_0<multiply_unlike*R)
            distance_local_1 = np.logical_and(distance_1>0,(distance_1*constant)<multiply_like*R)
    
            neighbourhood_0 = np.where(distance_local_0)
            #print("Raw Indices: ", neighbourhood_0)
            list_neighbours_0 = list(neighbourhood_0[0])
            neighbours_0 = [type_0[m] for m in list_neighbours_0]
            #print(list_neighbours_0)
            
            neighbourhood_1 = np.where(distance_local_1)
            #print("Raw Indices: ", neighbourhood_1)
            list_neighbours_1 = list(neighbourhood_1[0])
            neighbours_1 = [type_1[n] for n in list_neighbours_1]
            #print(list_neighbours_1)
            
        #print(like_0)
        #print(like_1)
        
        x_dist_0 = delta_x0[distance_local_0]
        y_dist_0 = delta_y0[distance_local_0]    
        positions_0 = np.array([x_dist_0,y_dist_0])
        
        x_dist_1 = delta_x1[distance_local_1]
        y_dist_1 = delta_y1[distance_local_1]
        positions_1 = np.array([x_dist_1,y_dist_1])
        
        #print("Neighbourhood Distances 0: ", positions_0)
        #print("Neighbourhood Distances 1: ", positions_1)
        #print("Neighbourhood Distance Magnitude 0: ", np.sqrt(np.square(positions_0[0])+np.square(positions_0[1])))
        #print("Neighbourhood Distance Magnitude 1: ", np.sqrt(np.square(positions_1[0])+np.square(positions_1[1])))
        force_0 = artificial_gravity(neighbours_0,positions_0,robot,R,max_force,like_0,max_force_overall,scale)
        force_1 = artificial_gravity(neighbours_1,positions_1,robot,R,max_force,like_1,max_force_overall,scale)
        
        
        
        friction = random.uniform(-0.1,0.1)
        force = force_0 + force_1 #+ friction

        #calculating the change in velocity
        delta_vx = force[0]*timestep/robot.mass
        delta_vy = force[1]*timestep/robot.mass

        #calculating the new velocity
        velocity_x = robot.x_velocity + delta_vx#velocity_coefficient*
        velocity_y = robot.y_velocity + delta_vy#velocity_coefficient*        
    
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

        
        
        #print("Finished 1 Loop within Physics Walk")
    #print("Finished the overall Physics Walk Calculation")


# In[8]:


#Function to calculate artificial gravity forces
def artificial_gravity(neighbourhood,positions, robot, R, max_force,like,max_force_overall,scale):
    
        # print("R: ",R)
        # print("Constant: ", constant)
        force = np.array([0,0])
        force_change_dir = 0
        #print(neighbours)
        #print("Artificial Gravity Neighbours: ", neighbours)
        #print("Artificial Gravity Neighbours Magnitudes: ", (np.sqrt(np.square(neighbours[0])+np.square(neighbours[1]))))
        # print(R*1.7)
        loop_counter = 0
        
        for x,y in zip(positions[0],positions[1]):
            G = robot.G
            constant = 1/robot.lattice
                
            object_position = np.array([(neighbourhood[loop_counter].x-robot.x),(neighbourhood[loop_counter].y-robot.y)])
            
            #print("Object Distance: ", np.array([delta_x,delta_y]))
            #distance_0 = np.sqrt(np.square(delta_x)+np.square(delta_y))
            
            #neighbour = neighbours[i]
            #print(i)
            #storing current position 
            current_position = np.array([x,y])
            """if(not np.array_equal(current_position,object_position)): 
                print("Object Distance: ", object_position)
                print("Neighbourhood Distance: ", current_position)
            """
            #calculating magnitude of positions
            mag_unshifted = (magnitude(current_position[0],current_position[1]))
            """#print(mag_unshifted)
            if(mag_unshifted > R):
                #print("Larger Distance Value: ", mag_unshifted)
                if(mag_unshifted>1.7*R):
                    #print("Error! Distance is too large. Value is ",mag_unshifted)"""
            if(like):
                mag = mag_unshifted*constant
            else:
                mag = mag_unshifted

            
            #calculating the force
            numerator = (robot.G*(robot.mass**2))
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
            force_delta = (force_change_dir*force_change)/scale

            #constraining force to the maximum
            if((force_delta[0]**2+force_delta[1]**2)>((max_force)**2)):
                #calculating unit vector of the force
                unit_force = unit([force_delta[0],force_delta[1]])
                
                #multiplying it by the maximum force
                updating_force = unit_force*max_force
                
                #setting the new force equal to the respective updated maximum force components
                force_delta[0] = updating_force[0]
                force_delta[1] = updating_force[1]
            force = force+force_delta
            loop_counter = loop_counter + 1
            
        #setting force maximum for sum of force contributions
        if((force[0]**2+force[1]**2)>((max_force_overall)**2)):
                #calculating unit vector of the force
                unit_force = unit([force[0],force[1]])
                
                #multiplying it by the maximum force
                updating_force = unit_force*max_force
                
                #setting the new force equal to the respective updated maximum force components
                force[0] = updating_force[0]
                force[1] = updating_force[1]
        #print("Finished")
        return force
    


# In[9]:


#Function to calculate distance between a robot and the swarm
def dist(robot, x,y):
    #finding the distance between the current robot and the other robots
    x_dist = x-robot.x
    y_dist = y-robot.y
    return x_dist,y_dist


# In[10]:


#Function to split swarm into different types
def split(swarm):
    type_0 = []
    type_1 = []
    for robot in swarm:
        if(robot.identifier == 0):
            type_0.append(robot)
        if(robot.identifier == 1):
            type_1.append(robot)

            
    return type_0,type_1
    


# In[11]:


def broadcast(swarm, detection_list, set_R):
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
    
            


# In[12]:


#Function to calculate the G transition parameter
def G_square(max_force,R,power):
    '''
    Calculates the transition value between liquid and solid states for the G parameter

            Parameters:
                   max_force (float): the maximum float limit on the force
                   R (float): the chosen R seperation float value
                   power (float): the chosen power for the artificial gravity equation
                   
            Returns:
                   G_transition (float): the G transition value for the current swarm setup

    '''
    G_transition = (max_force*(R**power))/(2+2*math.sqrt(2))
    return G_transition


# In[13]:


def retrieve_types(swarm):
    types = []
    for robot in swarm:
        types.append(robot.identifier)
    return types


# ## Physics Constants

# In[14]:


def simulation_run(run_number,runs,R,robot_number,max_force,base_path,image_path,time,probability_dust,bound,load_path):

    global grid_results
    global dust_measurement_result
    global dust_detection_result
    
    

    #time = 5000
    
 
    #print(robot_number)
    #print(max_force)
    #setting physics constants
    power = 2

  

    multiply = 1.5

    #initialising parameters
    conversion_ms = 1/3.6 #km/h to m/s conversion actor
    
    #Robot Parameters
    robot_speed = 2 #m/s
    robot_radius = 1 
    robot_mass = 1 #kgfro
    x_robot = []
    y_robot = []
    min_neighbour = []
    gen_random = 0 #the random angle

    #dust devil parameters
    dust_time = 170 #time in seconds the dust devil will survive 
    dust_speed = 5 #m/s
    dust_radius = 6
    dust_devils = []
    detection = 100

    



    lattice_constants = [math.sqrt(3),math.sqrt(2)]
    initialise_array = [20]
    lattice = lattice_constants[0]
    honeycomb = True
    #robot_number = 100#100
    


    #robot_speed_array = [1,2,3,4,5,6,7,8,9,10]
    
    max_force_overall = 99
    velocity_coefficient_array = [0.5,0.1,0.9]

    path = base_path 
    try:
        os.mkdir(path)
    except OSError:
        print ("Failure: Directory creation of %s failed" % path)
    else:
        print ("Success: Directory creation of %s succeeded" % path)

    image = (image_path + "Images/")
    try:
        os.mkdir(image)
    except OSError:
        print ("Failure: Directory creation of %s failed" % image)
    else:
        print ("Success: Directory creation of %s succeeded" % image)
        

    if(runs == 1):
        code = str(robot_number) + " Robots"
    else:
        code = "Run_" + str(run_number)
    
    load_path_inner = load_path + str(robot_number) + " Robots/" + "Run_" + str(run_number) + "/"

    values = []
    timestep = 1
    frequency = 1

    #R_array = [20,50,100,150,200,250,500,1000]#[10,20,50,100]
    #R_array = [50,60,75,90,100]
    grid = np.array([-bound,bound])

    grid_length = 10
    grid_performance = []


    velocity_coefficient = velocity_coefficient_array[0]

    G_state = G_transition(max_force,R,power)
   

    
    random_seed = random.randint(0,10000)
    random.seed(random_seed)#random_seed)

    performance_metric = []





    #for initial in initial_array:

    #only set seed if eliminating bias
    #random_seed = os.urandom(8)
    #random.seed(random_seed)
    #setting up the robot setup
    if(load_path == ""):
        print("Yes")
        swarm = initialise(robot_number, side, initialise_array[0],robot_speed,R,random_seed, lattice_constants,G_state)
    else:
        load_path = load_path + str(robot_number) + "_Robots/"
        types_array = np.load(load_path_inner+"Robot Types.npy")
        X,Y = load_positions((load_path_inner+"Robots.npy"),5000-1)
        swarm = pre_initialise_types(X,Y,robot_speed,R, types_array,lattice_constants,G_state)
    types_array = np.array(retrieve_types(swarm))
    
    store_robots = np.zeros((2,robot_number,(time*timestep)))
    store_dust = np.zeros((2,robot_number,(time*timestep)))
    min_neighbours = []
    cluster_average = []
    store_dust = []
    dust_devils = []
    measurement_events = 0
    detection_metric = 0
    dust_count = 0 
    total_detection = []
    total_collision = []
    total_dust = []

   
    starting_position_robots = positions(swarm)
    np.save(path + "Swarm Starting Position.npy",starting_position_robots)
    
    reduced_R = 20
    decay_time = 5
    
    for j in range(0,time*frequency):
        #updates dust devils positions, done at start of timestep, so all of the dust devil positions are recorded
        update_dust(dust_devils,bound)
        
        update_decay(swarm,R,G_state)
        #updating robots and storing robot/dust devil positions

        

        #checking if 1 second has passed and updating velocities of the robots via the physics walk algorithm and updating status of dust devils
        if(j%frequency==0):
            dust_count = dust_count + dust(dust_devils,probability_dust,side,j,dust_speed,dust_time,timestep,frequency,bound)
            physics_walk_honey(swarm,power,max_force,multiply,1,lattice,velocity_coefficient,max_force_overall)    

        #calling update timestep method which updates positions of robots, and returns performance metrics
        measurement_events_change,detection_metric_change = update_timestep(swarm,dust_devils,timestep,frequency,min_neighbours,cluster_average,detection,R,multiply,reduced_R,decay_time,bound)
        measurement_events = measurement_events + measurement_events_change
        detection_metric = detection_metric + detection_metric_change

        #appending performance metrics into list for storage/post processing
        total_detection.append(detection_metric)
        total_collision.append(measurement_events)

        

        #storing dust devil positions
        store_dust.append(positions(dust_devils))
        
        #storing the number of dust devils generated
        total_dust.append(dust_count)
        store_robots[:,:,j] = positions(swarm)
        
    type_0_object,type_1_object = split(swarm)
    x_0,y_0 = positions(type_0_object)
    x_1,y_1 = positions(type_1_object)

    retrieve_types
    
    print(store_robots[:,0])
    print(store_robots[:,].size)
    current_grid_metric = grid_coverage(grid,store_robots[:,:,time*frequency-1],grid_length)
    #saving robot positions
    np.save(path + "Robots.npy",store_robots)

    #saving dust devil positions
    with open( path + "dust.txt", "w") as f:
        json.dump(store_dust, f)

    #saving performance metrics - these could not be stored in excel due to storage limits
    np.save(path + "Minimum Distance to Neighbours.npy",np.array(min_neighbours))
    np.save(path + "Cluster Average.npy", np.array(cluster_average))
    np.save(path + "Measurement Events Count.npy", np.array(total_collision))
    np.save(path + "Number of Dust Devils Detected.npy", np.array(total_detection))
    np.save(path + "Number of Dust Devils Generated.npy", np.array(total_dust))

    np.save(path + "Robot Types.npy",types_array)
    
    count_generated_dust = total_dust[time-1]
    count_detected_dust = total_detection[time-1]
    count_measurements_dust = total_collision[time-1]

    #adding values tested for a values excel spreadsheet for easy viewing
    values.append([len(swarm),timestep,G_state,power,R,multiply,max_force,robot_speed,min_neighbours[time*frequency-1],cluster_average[time*frequency-1],random_seed, G_state,measurement_events,detection_metric,dust_count])

    #the titles of the variables in the values array
    values_title = ['Number of Robots',
                            'Timestep',
                            'G', 
                            'Power', 
                            'Communication Range', 
                            'Multiplier',
                            'Max Force',
                            'Max Speed',
                            'Minimum Distance to Neighbours',
                            'Average Cluster Size',
                            'Seed',
                            'G Transition',
                            'Measurement Events Count',
                            'Number of Dust Devils Detected',
                            'Total Number of Dust Devils',
        ]

    #storing the constants used in the simulation
    data = {'Values':[len(swarm),timestep,power,R,multiply,max_force,robot_speed,random_seed, G_state]}
    titles =['Number of Robots',
                            'Timestep', 
                            'Power',
                            'Communication Range', 
                            'Multiplier',
                            'Max Force',
                            'Max Speed',
                            'Seed',
                            'G',
        ]
    #setting the constants as a panda dataframe
    constants = pd.DataFrame(data, index = titles
                           ) 

    constants.to_excel(path + "Constants.xlsx")
    #print("G_state: ",G_state)
    #print("G_parameter: " ,G_state)

    """
    #setting the start paths for the graphs and the tables
    graph_start_path = image + code + " - Graph_Beginning.png"
    table_start_path = image +code + " - Table_Beginning.png"

    #using the processing functions to create plotly graphs and tables for the figures in the first timestep
    graph_start = Processing_Functions_Tracking.graph_figure(store_robots,0,frequency,code)
    graph_start.write_image(graph_start_path)
    table_start = Processing_Functions_Tracking.table_figure(store_robots,0,frequency,constants,min_neighbours,cluster_average,total_collision,total_detection,total_dust)
    table_start.write_image(table_start_path)

    #combining the tables and the graphs using pillow
    Processing_Functions_Tracking.combine_tracking(graph_start_path,table_start_path)
    #setting the end paths for the graphs and the tables
    graph_end_path = image + code + " - Graph_End.png"
    table_end_path =image + code + " - Table_End.png"

    #using the processing functions to create plotly graphs and tables for the figures in the last timestep
    graph_end = Processing_Functions_Tracking.graph_figure(store_robots,time-1,frequency,code)
    graph_end.write_image(image + code + " - Graph_End.png")
    table_end = Processing_Functions_Tracking.table_figure(store_robots,time-1,frequency,constants,min_neighbours,cluster_average,total_collision,total_detection,total_dust)
    table_end.write_image(image + code + " - Table_End.png")
    
    #combining the tables and the graphs using pillow
    Processing_Functions_Tracking.combine_tracking(graph_end_path,table_end_path)
    
    #using the processing functions to create plotly graphs and tables for the figures in the last timestep
    graph_end_types = Processing_Functions_Tracking.graph_types(x_0,y_0,x_1,y_1,bound,10,"Deployed Swarm Formation ", " <b>Timestep = " + str(time) + " s<br>R = " + str(round(R,2)) + "</b> <br> ")
    graph_end_types.write_image(image + code + " - Graph_End_Types.png")



    #using the processing functions to create plotly graphs and tables for the figures in the last timestep
    graph_end_area_coverage = Processing_Functions_Tracking.graph_area_coverage(x_0,y_0,x_1,y_1,bound,10, "Area Coverage Over a Grid for a Deployed Swarm Formation"," <b>Timestep = " + str(time) + " s<br>R = " + str(round(R,2)) + "<br>Area Coverage = " + str(current_grid_metric) + "%</b><br> ")
    graph_end_area_coverage.write_image(image + code + " - Graph_End_Area_Coverage.png")


    #plotting performance of the average of minimum neighbouring distance metric
    performance = Processing_Functions_Tracking.performance_graph(min_neighbours,np.linspace(0,len(min_neighbours),len(min_neighbours)*frequency,endpoint = False),frequency,code,"Time (s)","Minimum Average Neighbour Distance (m)")
    performance.write_image(image + code + " - Minimum Neighbour Average.png")

    #plotting performance of the dust devil measurement metric
    performance_measurement = Processing_Functions_Tracking.performance_graph(total_collision,np.linspace(0,len(total_collision),len(total_collision)*frequency,endpoint = False),frequency,code,"Time (s)","Count of Measurement Events")
    performance_measurement.write_image(image + "/" + code + " - Count of Measurements in Dust Devil.png")
    
    detection_code = code + " and " + str(total_dust[time*frequency-1]) +" Dust Devils"
    #plotting performance of the dust devil detection metric
    performance_detection = Processing_Functions_Tracking.performance_graph(total_detection,np.linspace(0,len(total_detection),len(total_detection)*frequency,endpoint = False),frequency,detection_code,"Time (s)","Count of Dust Devils Detected")
    performance_detection.write_image(image + "/" + code + " - Number of Dust Devils Detected.png")

    #plotting performance of the cluster average of the swarm
    cluster = Processing_Functions_Tracking.performance_graph(cluster_average,np.linspace(0,len(cluster_average),len(cluster_average)*frequency,endpoint = False),frequency,code,"Time (s)","Average Cluster Size")
    cluster.write_image(image + code +  " - Average Cluster Size.png")
    """

    #setting panda dataframe for the values tested
    df = pd.DataFrame(values, columns = values_title)

    #saving at as an excel spreadsheet
    df.to_excel(path + "Results.xlsx")


    #performance_metric_np = np.array(performance_metric)

    #saving performance metrics
    #np.save(original_path + "Performance Metric " + str(timestep) + ".npy", performance_metric_np)
    """
    #plotting performance of the different timesteps
    #performance_overall = Processing_Functions_BP.performance_graph(performance_metric_np[:,1],performance_metric_np[:,0],1,code,"Number of Timesteps in 1 Second","Number of Dust Devils Detected")
    #performance_overall.write_image(new_path + "Overall Performance.png")
    grid_performance.append([current_grid_metric,R])

    """
       # performance_metric.append([robot,total_detection[-1]])
   

    performance_metric_np = np.array(grid_performance)
    np.save(path + code + " Grid Performance Metric.npy", performance_metric_np)
    #performance_overall = Processing_Functions_Tracking.performance_graph(performance_metric_np[:,0],performance_metric_np[:,1],1,code,"Desired Seperation Distance, R (m) ","Grid Coverage Percentage")
    #performance_overall.write_image(original_path + "Overall Performance.png")
    grid_metric_data = [R,current_grid_metric]
    grid_results.append(grid_metric_data)
    #print(100-current_grid_metric)
    x_max = max(abs(store_robots[0,:,time*frequency-1]))
    y_max = max(abs(store_robots[1,:,time*frequency-1]))
    return grid_metric_data,count_generated_dust,count_detected_dust,count_measurements_dust


# In[15]:


def parameter_sweep(i,path,runs,time,bound,probability_dust,load_path):

    robot_array = [10,20,30,40,50,60,70,80,90,100]
    max_force_array = [100,50,50,100,100,25,100,25,25,100]
    R_array = [220.97977053140096,
               168.6150512214342, 
               121.75360082304526,
               122.83730158730158,
               119.0753690753691,
               117.63988808952837,
               119.46217494089835,
               81.81289123630673,
               80.28360528360528,
               81.1255787037037
                ]
    
    
    robot_number = robot_array[i]
    max_force = max_force_array[i]
    R=R_array[i]
    
    
    outer_path = path + str(robot_number) + " Robots/"
    
    try:
        os.mkdir(outer_path)
    except OSError:
        print ("Failure: Directory creation of %s failed" % outer_path)
    else:
        print ("Success: Directory creation of %s succeeded" % outer_path)

    image_path = outer_path

    global grid_results
    grid_results = []

    
    #grid metric data manipulation
    calculating_average = []
    optimised_area_results = []
    calculating_area_coverage = []
    count_generated_dust_array = []
    count_detected_dust_array = []
    count_measurements_dust_array = []
    
    for run_count in range(runs):
        if(runs == 1):
            base_path = outer_path
            image_path = path
        else:
            base_path = outer_path + "Run_" + str(run_count) + "/"
            try:
                os.mkdir(base_path)
            except OSError:
                print ("Failure: Directory creation of %s failed" % base_path)
            else:
                print ("Success: Directory creation of %s succeeded" % base_path)
            

        grid_return,count_generated_dust,count_detected_dust,count_measurements_dust = simulation_run(run_count,runs,R,robot_number,max_force,base_path,image_path,time,probability_dust,bound,load_path)
        count_generated_dust_array.append(count_generated_dust)
        count_detected_dust_array.append(count_detected_dust)
        count_measurements_dust_array.append(count_measurements_dust)
    
        
        #rest of the inner loop deals with grid metric data manipulation for easy viewing in the data
        results_sorted = grid_return
       
        add = grid_return
        
        
        optimised_area_results.append(add)
        calculating_average.append(R)
        calculating_area_coverage.append(add[1])
        temp = np.array(grid_results)
    if(runs>1):
        #averages
        average_array = np.array(calculating_average)

        generated_np = np.array(count_generated_dust_array)
        detected_np = np.array(count_detected_dust_array)
        measurement_np = np.array(count_measurements_dust_array)
        np.save(outer_path+"Array of Final Number of Generated Dust Devils.npy", generated_np)
        np.save(outer_path + "Array of Final Number of Detected Dust Devils.npy", detected_np)
        np.save(outer_path + "Array of Final Count of Dust Devil Measurements", measurement_np)

        average_generated = np.mean(generated_np)
        average_detected = np.mean(detected_np)
        average_measurement = np.mean(measurement_np)

        std_generated = np.std(generated_np)
        std_detected = np.std(detected_np)
        std_measurement = np.std(measurement_np)

        result_generated = np.array([average_generated, std_generated])
        result_detected = np.array([average_detected, std_detected])
        result_measurement = np.array([average_measurement, std_measurement])

        np.savetxt(outer_path + str(robot_number) + ' Robots - Number of Dust Devils Generated.txt', result_generated)
        np.savetxt(outer_path + str(robot_number) + ' Robots - Number of Dust Devils Detected.txt', result_detected)
        np.savetxt(outer_path + str(robot_number) + ' Robots - Count of Measurements Taken Within Dust Devils.txt', result_measurement)
        file_generated = open(path + 'Number of Dust Devils Generated vs vs Number of Robots.txt', 'a')
        file_detected = open(path + 'Dust Detection Metric vs Number of Robots.txt.txt', 'a')
        file_measurement = open(path + 'Dust Measurement Metric vs Number of Robots.txt.txt', 'a')


        file_generated.write(str([robot_number,result_generated]))
        file_detected.write(str([robot_number,result_detected]))
        file_measurement.write(str([robot_number,result_measurement]))

        file_generated.close()
        file_detected.close()
        file_measurement.close()

        #Grid Coverage Results
        np.save(outer_path+"Area Coverage Result.npy",np.array(optimised_area_results))
        average_array = np.array(calculating_average)
        average_area_coverage = np.mean(np.array(calculating_area_coverage))
        std_area = np.std(np.array(calculating_area_coverage))
        area_results = np.array([average_area_coverage,std_area])
        #average = np.mean(average_array)
        #std_R = np.std(average)
        #average_result = np.array([average,std_R])
        #np.save(outer_path+"Averaged R Value.npy",average_result)
        #print("Average R Results: ",average_result)
        #np.savetxt(outer_path + str(robot_number) + ' Robots R Average Result.txt', average_result)
        np.savetxt(outer_path + str(robot_number) + ' Robots % Coverage Average Result.txt', area_results)

        #return result_generated,result_detected,result_measurement


# In[ ]:


import sys
global grid_results
from concurrent import futures


i = int(sys.argv[1]) 


#num_cores = multiprocessing.cpu_count()
robot_array = [10,20,30,40,50,60,70,80,90,100]

number_of_runs = 10#1
bound = 500
time = 100
global grid_results

print("Running..")
#Area Parameters
side = 1 #each side length in m
per_square_km = 100/(60*60*24)
probability_dust = per_square_km#(per_square_km*area) #number of dust devils per second


grid_results = []
path = "Experiments/Static_Trial/"
load_path = "Experiments/Initial/"#../Experiments/Detecting Dust Devils - Passive Survey 1. Final Deployment Sweep with Dust Devils and Multiple Runs/"
try:
    os.mkdir(path)
except OSError:
    print ("Failure: Directory creation of %s failed" % path)
else:
    print ("Success: Directory creation of %s succeeded" % path)
    



#parameter_sweep(3,path,number_of_runs,time,bound,probability_dust,file_generated,file_detected,file_measurement)
parameter_sweep(i,path,number_of_runs,time,bound,probability_dust,load_path)
#results = Parallel(n_jobs=6)(delayed(parameter_sweep)(i,path,number_of_runs,time,bound,probability_dust,load_path) for i in range(len(robot_array)))



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




