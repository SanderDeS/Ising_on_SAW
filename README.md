# Ising_on_SAW
This program simulates an Ising model on a SAW using the metropolis algorithm.
This repository contains a python file and a file for jupyter-lab or jupyter-notebook.

This program takes some time to run dependent on the length of the self-avoiding walk 
and the number of sweeps for the Monte-Carlo simulation. 

The simulation makes one move every sweep. This move can be a spin flip or a change in the path.
A change in path is possible for all points except the begin- and endpoint. Although the functions 
to move these two points are present, there is still a bug somewhere in the code that needs to be and will be fixed. 
