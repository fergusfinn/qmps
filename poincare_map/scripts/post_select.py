#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from argparse import ArgumentParser
TwoBodySim = __import__('2body_scars')
import yaml 
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser(description="Post Select on Energy for quantum scars")
    parser.add_argument('input_folder')
    parser.add_argument('output_name')
    arguments = parser.parse_args()
    
    # loop through files in input directory, get the answers and 
    #   add in a single numpy array
    
    input_files = os.getcwd() + "/" + arguments.input_folder
    
    directory = os.fsencode(input_files)
    
    all_angles = np.empty((0,4))
    
    for file in os.listdir(directory)[:10]:
        with open(os.path.join(directory, file)) as f:
            data = yaml.load(f , Loader = yaml.Loader)
        
        angles = data['data']
        poincare_angles = TwoBodySim.poincare_section(angles = angles, t = range(len(angles)))
        all_angles = np.concatenate((all_angles, poincare_angles), axis = 0)
    
    
    # apply energy bounds to keep the best 10% of points in each quadrant
    phi1_line = 1.5
    theta2_line = 5.6
    
    # divide the angles into quadrants:
    """
    ___ ___
   |q2 |q4 |
   |---|---|
   |q1_|q3_|
        
    """
    q1 = all_angles[np.logical_and(all_angles[:,1] < phi1_line, all_angles[:,3] < theta2_line)]
    q2 = all_angles[np.logical_and(all_angles[:,1] < phi1_line, all_angles[:,3] > theta2_line)]
    q3 = all_angles[np.logical_and(all_angles[:,1] > phi1_line, all_angles[:,3] < theta2_line)]
    q4 = all_angles[np.logical_and(all_angles[:,1] > phi1_line, all_angles[:,3] > theta2_line)]
    
    all_qs = [q1,q2,q3,q4]
    
    # loop through the quadrants and keep the 10% of energues closest to 
    #   0 energy
    post_selected = np.empty((0,4))
    percentage_allowed = 0.1
    for q in all_qs:
        energy = np.abs(TwoBodySim.apply_energy(q))
        energy_rank = energy.argsort().argsort() / len(energy)
        post_selected = np.concatenate((post_selected, q[energy_rank < percentage_allowed])) 
        
    post_selected[:,0] = range(1, post_selected.shape[0]+1)
    # save post-selected energies in a txt file
    output_folder = os.getcwd() + "/inputs/"
    
    fmt = '%d', '%1.15f', '%1.15f', '%1.15f'
    np.savetxt(output_folder + arguments.output_name, post_selected, fmt = fmt)
    
    
    
    
    
    
    
    
    
    
    
