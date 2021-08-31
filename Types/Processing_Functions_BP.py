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



import random
import datetime
import os
import math
from ast import literal_eval
from PIL import Image

#Function to check if any of the items contain an array
def array_check(item,timer):
    '''
    Used to evaluate a number of different data types
    
            Parameters:
                   item (tuple object): a tuple object

                   
            Returns:
                   data (float): the data being returned in a readable float form
    '''
    #indices used to retrieve data from tuple object
    item = item[0]

    #check if it is a string
    if(isinstance(item, str)):
        item = literal_eval(item)
        data = round(item[timer],2)
        return data
    
    #check if it is a list, and return the last item on the list if so
    elif(isinstance(item, list)):
        data = round(item[timer],2)
        return data
    else:
        data = round(float(item),2)
        return data
        
def graph_figure(robots,timer,frequency,code):
    #initialising global variables
    

    position_robot = robots[:,:,timer*frequency]
    x_max = max(abs(robots[0,:,timer]))
    x_mag = math.floor(math.log((x_max),10))
    y_max = max(abs(robots[1,:,timer]))

    
    y_mag = math.floor(math.log((y_max),10))
    if(x_max > 1000):
        x_max = math.ceil(x_max/(10**x_mag))*(10**x_mag)
    if(y_max > 1000):
        y_max = math.ceil(y_max/(10**y_mag))*(10**y_mag)
    if(x_max < 1000):
        x_max = 1000
    if(y_max < 1000):
        y_max = 1000
    x_max = x_max/frequency
    #creating scatter plot of robots
    data = go.Scatter(
        x=list(position_robot[0]/frequency),
        y=list(position_robot[1]),
        name = 'Robots',
        mode = 'markers',
    )
    x_max = 2000
    y_max = 2000
    
    #creating the plotly figure with the robot data
    fig = go.Figure(
        { "data": data , "layout": go.Layout(yaxis=dict(range=[-y_max, y_max]),xaxis = dict(range=[-x_max,x_max]))
        }
    )

    #updating layout with circles and different formatting'''
    fig.update_layout(title="<b>Physics Based Swarm Experiment "+ code + "</b>",
    title_x=0.5,
    xaxis_title="X Position (m)",
    yaxis_title="Y Position (m)",
    margin=dict(
        t=50, # top margin: 30px, you want to leave around 30 pixels to
              # display the modebar above the graph.
         # bottom margin: 10px
        l=10, # left margin: 10px
        r=10, # right margin: 10px
    ),

#     height=900,width=1150,
     xaxis = dict(
         tickmode = 'linear',
         tick0 = 0,
         dtick = x_max/10
     ),
                       yaxis = dict(
         tickmode = 'linear',
         tick0 = 0,
         dtick = y_max/10
     )
                     )
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black',mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black',mirror=True)
    
    return fig
    


def performance_graph(fitness,x_values,frequency,code,x_title,y_title):
    #initialising global variables
    print(code)
    if(code == ""):
        code = ""
    else:
        code = " with " + code
    
    #creating scatter plot of robots
    data = go.Scatter(
        x=x_values/frequency,
        y=fitness,
        mode = 'lines+markers',
    )
    
    #creating the plotly figure with the robot data
    fig = go.Figure(
        { "data": data
        }
    )
    
    #updating layout with circles and different formatting'''
    fig.update_layout(title="<b>" + y_title + " versus " + x_title + " for Experiment" + code + "</b>",
    title_x=0.5,
    xaxis_title=x_title,
    yaxis_title=y_title,
    margin=dict(
        t=50, # top margin: 30px, you want to leave around 30 pixels to
              # display the modebar above the graph.
         # bottom margin: 10px
        l=10, # left margin: 10px
        r=10, # right margin: 10px
    ),
    height=900,width=1150,
                      yaxis = {'range': [0, math.ceil(np.max(fitness)+1)],}
                            
    
                     )
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black',mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black',mirror=True)
    
    return fig

def multiple_graphs(performance_array,timestep,code,x_title,y_title,labels):
    print(performance_array)
    print(len(performance_array))
    for i in range(len(performane_array)):
        values = array[i]
   
        np_array = np.array(values)
        print(np_array)
        print("First" + str(np_array[:,0]))
        data.append(go.Scatter(
        x=np_array[:,0],
        y=np_array[:,1],
        mode = 'lines+markers',
        name = labels[i]))
    
    layout = go.Layout(title="<b>" + y_title + " versus " + x_title + " for "+ code + " Experiment</b>",
    title_x=0.5,
    xaxis_title=x_title,
    yaxis_title=y_title,
    margin=dict(
        t=50, # top margin: 30px, you want to leave around 30 pixels to
              # display the modebar above the graph.
         # bottom margin: 10px
        l=10, # left margin: 10px
        r=10, # right margin: 10px
    ),
    height=900,width=1150,
                      yaxis = {'range': [0, math.ceil(np.max(fitness)+1)],}
                            )

    fig = go.Figure(data=data, layout=layout)
 
    
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black',mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black',mirror=True)
    
    return fig
 
def table_figure(robots,timer,frequency,constants,min_neighbours,cluster_average,total_collisions,total_detection,total_dust):
    fig = go.Figure(data=[go.Table(header=dict(values=['<b>Simulation Parameters</b>','<b>Values</b>'],
                            line_color='black',
                            font=dict(color='black', size=17)),
                            cells=dict(values=[[ '<b>Simulation Timestep (h:min:s)</b>','<b>Number of Robots</b>','<b>Timestep Size (s) </b>','<b>Communication Range (m)</b>', '<b>Gravitational Constant</b>','<b>Power</b>','<b>Local Multiplier</b>','<b>Max Force (N)</b>', '<b>Max Speed (m/s)</b>', '<b>Minimum Neighbour Average (m)</b>', '<b>Average Cluster Size</b>','<b>Measurement Events Count </b>', '<b>Number of Dust Devils Detected </b>', '<b>Total Number of Dust Devils</b>'], [str(datetime.timedelta(seconds=timer)),len(robots[0,:,timer*frequency]),round(1/frequency,2),array_check(constants.loc["Communication Range"].values,timer*frequency),array_check(constants.loc["G"].values,timer*frequency),array_check(constants.loc["Power"].values,timer*frequency),array_check(constants.loc["Multiplier"].values,timer*frequency),array_check(constants.loc["Max Force"].values,timer*frequency),array_check(constants.loc["Max Speed"].values,timer*frequency),round(min_neighbours[timer*frequency],2),cluster_average[timer*frequency],total_collisions[timer*frequency],round(total_detection[timer*frequency]),total_dust[timer*frequency]]],align='center',line_color='black'))])
    fig.update_layout(width = 650, height = 800,margin=dict(
t=260, # top margin: 30px, you want to leave around 30 pixels to
              # display the modebar above the graph.
        b=10, # bottom margin: 10px
        l=10, # left margin: 10px
        r=10,
    ))
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    return fig

def table_figure_area(robots,timer,frequency,constants,min_neighbours,cluster_average):
    fig = go.Figure(data=[go.Table(header=dict(values=['<b>Simulation Parameters</b>','<b>Values</b>'],
                            line_color='black',
                            font=dict(color='black', size=17)),
                            cells=dict(values=[[ '<b>Simulation Timestep (h:min:s)</b>','<b>Number of Robots</b>','<b>Timestep Size (s) </b>','<b>Communication Range (m)</b>', '<b>Gravitational Constant</b>','<b>Power</b>','<b>Local Multiplier</b>','<b>Max Force (N)</b>', '<b>Max Speed (m/s)</b>', '<b>Minimum Neighbour Average (m)</b>', '<b>Average Cluster Size</b>'], [str(datetime.timedelta(seconds=timer)),len(robots[0,:,timer*frequency]),round(1/frequency,2),array_check(constants.loc["Communication Range"].values,timer*frequency),array_check(constants.loc["G"].values,timer*frequency),array_check(constants.loc["Power"].values,timer*frequency),array_check(constants.loc["Multiplier"].values,timer*frequency),array_check(constants.loc["Max Force"].values,timer*frequency),array_check(constants.loc["Max Speed"].values,timer*frequency),round(min_neighbours[timer*frequency],2),cluster_average[timer*frequency]]],align='center',line_color='black'))])
    fig.update_layout(width = 650, height = 800,margin=dict(
t=260, # top margin: 30px, you want to leave around 30 pixels to
              # display the modebar above the graph.
        b=10, # bottom margin: 10px
        l=10, # left margin: 10px
        r=10,
    ))
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    return fig

#function to combine the table and graph images
def combine(graph_path,table_path):
    image_graph = Image.open(graph_path)
    image_table = Image.open(table_path)
    new1 = Image.new('RGB', (image_graph.width + image_table.width, min(image_graph.height,image_table.height)))
    new1.paste(image_graph, (0, 0))
    new1.paste(image_table, (image_graph.width,((image_graph.height-image_table.height)//2)-100))
    new1.save(graph_path)
    os.remove(table_path)
