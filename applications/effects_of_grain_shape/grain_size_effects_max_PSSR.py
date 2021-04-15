import os
import numpy as np
import sys
import pandas as pd
import operator
import glob
import pickle as p
import csv
import shutil
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as plt_cols
import matplotlib.cm as cm
import re
import glob

# Users can simply execute this script as "python max_PSSR.py" in a command prompt window or by using an interactive python window.
# This script should be placed in the directory that contains the "NoGrainSize", "WithGrainSize," etc. directories that were downloaded from the Materials Commons

# Get name of directory that contains the downloaded data and this script
DIR_LOC = os.path.dirname(os.path.abspath(__file__))


def grain_size_paper_max_slip():

    # Function to create plots for grain size and grain shape manuscript
    # This first section plots the maximum plastic shear strain range from simulations with cubic textured microstructures
    
    cubic_directories = [os.path.join(DIR_LOC, r'NoGrainSize\Cube_texture\no_gs_cubic_AR_1'),
                         os.path.join(DIR_LOC, r'NoGrainSize\Cube_texture\no_gs_cubic_AR_6'),
                         os.path.join(DIR_LOC, r'WithGrainSize\Cube_texture\gs_cubic_AR_1'),
                         os.path.join(DIR_LOC, r'WithGrainSize\Cube_texture\gs_cubic_AR_6')]
    
    # Define directory to store plots
    store_dirr = r'G:\Grain_size_paper\plots'
    
    # Create directory to store plots if it does not exist
    if os.path.exists(store_dirr):
        os.chdir(store_dirr)
    else:
        os.makedirs(store_dirr)
        os.chdir(store_dirr)
    
    # Initialize array
    max_slip_per_int_all_cubic = []
    
    # Specify number of results files in each folder
    num_in_each = [1, 2, 1, 2]
    
    # Go to each directory and read data from files
    for ii, directory in enumerate(cubic_directories):
    
        for jj in range(num_in_each[ii]):
    
            print('In %s' % directory)
            
            # Specify files with simulations values at points of maximum compression and maximum tension
            dirr_max_comp = os.path.join(directory, 'Max_Comp_%d.csv' % jj)
            dirr_max_tens = os.path.join(directory, 'Max_Ten_%d.csv' % jj)

            # Read in data using pandas module
            aver_comp = pd.read_csv(dirr_max_comp, index_col = False)
            aver_tens = pd.read_csv(dirr_max_tens, index_col = False)
            # "index_col = False" means that the first column is NOT the index, which is the case in the quadrature output files here.

            # Sort data
            aver_comp_2_sorted = aver_comp.sort_values(['z','y','x'], ascending = [True, True, True])
            aver_tens_2_sorted = aver_tens.sort_values(['z','y','x'], ascending = [True, True, True])

            # Calculate 
            delta_gamma = (aver_tens_2_sorted - aver_comp_2_sorted) 
            slip_values = abs(delta_gamma[['slip_' + str(i) for i in range(1,13)]]).to_numpy()
            
            # Get the maximum plastic shear strain range from each integration point
            max_slip_per_int = np.max(slip_values, axis = 1)
            max_slip_per_int_all_cubic.append(max_slip_per_int)
        
    
    # Define limits for X axes
    max_slip_x_lims = [-0.1e-3, 5.5e-3]

    # Define plot characteristics
    labelss = ['AR 1', 'AR 6', r'AR $\frac{1}{6}$', 'AR 1 - GS', 'AR 6 - GS', r'AR $\frac{1}{6}$ - GS']
    cfm     = ['r', 'b', 'g', 'r', 'b', 'g']
    linestyless = ['-', '-', '-', ':', ':', ':']

    # Specify number of bins for histograms
    n_bins = 100
    
    # Plot figure
    fig = plt.figure(facecolor="white", figsize=(6, 4), dpi=1200)
    for rr, valss in enumerate(max_slip_per_int_all_cubic):
        
        n,x1,_ = plt.hist(valss, bins = n_bins, histtype = 'step', color = 'white')
        n1 = n.astype(int)
        bin_centers = 0.5*(x1[1:]+x1[:-1])
        plt.plot(bin_centers, n1, label = labelss[rr], color = cfm[rr], linestyle = linestyless[rr])    

    plt.legend(framealpha = 1, ncol = 2)
    plt.xlim(max_slip_x_lims[0], max_slip_x_lims[1])
    plt.ylim(0, 4.4e4)
    plt.ylabel('Num')   
    plt.grid(True)
    plt.xlabel('Maximum plastic shear strain range')     
    plt.ticklabel_format(style = 'sci', axis='both', scilimits=(0,0), useMathText = True)    
    plt.tight_layout()
    plt.savefig( os.path.join(store_dirr, 'cubic_all_%d_bins_v3' % n_bins) )
    plt.close()          


    # Plot a zoomed-in figure at the highest plastic shear strain range values
    n_bins = 100
    
    fig = plt.figure(facecolor="white", figsize=(4, 2.666), dpi=1200)
    for rr, valss in enumerate(max_slip_per_int_all_cubic):
        
        n,x1,_ = plt.hist(valss, bins = n_bins, histtype = 'step', color = 'white')
        n1 = n.astype(int)
        bin_centers = 0.5*(x1[1:]+x1[:-1])
        plt.plot(bin_centers, n1, label = labelss[rr], color = cfm[rr], linestyle = linestyless[rr])    

    # plt.show()
    plt.xlim(1.75e-3, 3e-3)
    plt.ylim(0, 100)
    plt.grid(True)
    plt.ticklabel_format(style = 'sci', axis='x', scilimits=(0,0), useMathText = True)    
    plt.tight_layout()
    plt.savefig( os.path.join(store_dirr, 'cubic_all_%d_bins_v3_ZOOM' % n_bins) )
    plt.close()          














    ''' Repeat for the random textured microstructures ''' 
    
    random_directories = [os.path.join(DIR_LOC, r'NoGrainSize\Random_texture\no_gs_random_AR_1'),
                          os.path.join(DIR_LOC, r'NoGrainSize\Random_texture\no_gs_random_AR_6'),
                          os.path.join(DIR_LOC, r'WithGrainSize\Random_texture\gs_random_AR_1'),
                          os.path.join(DIR_LOC, r'WithGrainSize\Random_texture\gs_random_AR_6')]
                   
    
    max_slip_per_int_all_random = []
    
    num_in_each = [1, 2, 1, 2]
    
    for ii, directory in enumerate(random_directories):
    
        for jj in range(num_in_each[ii]):
    
            print('In %s' % directory)
            
            # Specify files with simulations values at points of maximum compression and maximum tension
            dirr_max_comp = os.path.join(directory, 'Max_Comp_%d.csv' % jj)
            dirr_max_tens = os.path.join(directory, 'Max_Ten_%d.csv' % jj)

            # Read in data using pandas module
            aver_comp = pd.read_csv(dirr_max_comp, index_col = False)
            aver_tens = pd.read_csv(dirr_max_tens, index_col = False)
            # "index_col = False" means that the first column is NOT the index, which is the case in the quadrature output files here.

            aver_comp_2_sorted = aver_comp.sort_values(['z','y','x'], ascending = [True, True, True])
            aver_tens_2_sorted = aver_tens.sort_values(['z','y','x'], ascending = [True, True, True])


            delta_gamma = (aver_tens_2_sorted - aver_comp_2_sorted) 
            slip_values = abs(delta_gamma[['slip_' + str(i) for i in range(1,13)]]).to_numpy()
            
            max_slip_per_int = np.max(slip_values, axis = 1)
            max_slip_per_int_all_random.append(max_slip_per_int)
        
    
    # Define plot characteristics
    labelss = ['AR 1', 'AR 6', r'AR $\frac{1}{6}$', 'AR 1 - GS', 'AR 6 - GS', r'AR $\frac{1}{6}$ - GS']
    cfm     = ['r', 'b', 'g', 'r', 'b', 'g']
    linestyless = ['-', '-', '-', ':', ':', ':']
    

    n_bins = 100
    
    fig = plt.figure(facecolor="white", figsize=(6, 4), dpi=1200)
    for rr, valss in enumerate(max_slip_per_int_all_random):
        
        n,x1,_ = plt.hist(valss, bins = n_bins, histtype = 'step', color = 'white')
        n1 = n.astype(int)
        bin_centers = 0.5*(x1[1:]+x1[:-1])
        plt.plot(bin_centers, n1, label = labelss[rr], color = cfm[rr], linestyle = linestyless[rr])    

    # plt.show()
    plt.legend(framealpha = 1, ncol = 2)
    plt.xlim(max_slip_x_lims[0], max_slip_x_lims[1])
    plt.ylim(0, 4.4e4)
    plt.ylabel('Num')   
    plt.grid(True)
    plt.xlabel('Maximum plastic shear strain range')     
    plt.ticklabel_format(style = 'sci', axis='both', scilimits=(0,0), useMathText = True)    
    plt.tight_layout()
    plt.savefig( os.path.join(store_dirr, 'random_all_%d_bins_v3' % n_bins) )
    plt.close()   


    # Plot a zoomed-in figure at the highest plastic shear strain range values

    n_bins = 100
    
    fig = plt.figure(facecolor="white", figsize=(4, 2.666), dpi=1200)
    for rr, valss in enumerate(max_slip_per_int_all_random):
    # for rr in plt_order:
        
        n,x1,_ = plt.hist(valss, bins = n_bins, histtype = 'step', color = 'white')
        n1 = n.astype(int)
        bin_centers = 0.5*(x1[1:]+x1[:-1])
        plt.plot(bin_centers, n1, label = labelss[rr], color = cfm[rr], linestyle = linestyless[rr])    

    # plt.show()
    plt.xlim(max_slip_x_lims[0], max_slip_x_lims[1])
    plt.xlim(3e-3, max_slip_x_lims[1])
    plt.ylim(0, 50)
    plt.grid(True)
    plt.ticklabel_format(style = 'sci', axis='x', scilimits=(0,0), useMathText = True)    
    plt.tight_layout()
    plt.savefig( os.path.join(store_dirr, 'random_all_%d_bins_v3_ZOOM' % n_bins) )
    plt.close()   














    ''' Repeat for the rolled textured microstructures ''' 

    # Read ALL rolled data
    rolled_directories = [os.path.join(DIR_LOC, r'NoGrainSize\Rolling_texture\no_gs_rolled_AR_1'),
                          os.path.join(DIR_LOC, r'NoGrainSize\Rolling_texture\no_gs_rolled_AR_6'),
                          os.path.join(DIR_LOC, r'NoGrainSize\Rolling_texture\no_gs_rolled_AR_6c'),
                          os.path.join(DIR_LOC, r'WithGrainSize\Rolling_texture\gs_rolled_AR_1'),
                          os.path.join(DIR_LOC, r'WithGrainSize\Rolling_texture\gs_rolled_AR_6'),
                          os.path.join(DIR_LOC, r'WithGrainSize\Rolling_texture\gs_rolled_AR_6c')]
                   

    max_slip_per_int_all_rolled = []
    
    num_in_each = [2] * 6
    
    for ii, directory in enumerate(rolled_directories):
    
        for jj in range(num_in_each[ii]):
    
            print('In %s' % directory)
            
            # Specify files with simulations values at points of maximum compression and maximum tension
            dirr_max_comp = os.path.join(directory, 'Max_Comp_%d.csv' % jj)
            dirr_max_tens = os.path.join(directory, 'Max_Ten_%d.csv' % jj)

            # Read in data using pandas module
            aver_comp = pd.read_csv(dirr_max_comp, index_col = False)
            aver_tens = pd.read_csv(dirr_max_tens, index_col = False)
            # "index_col = False" means that the first column is NOT the index, which is the case in the quadrature output files here.

            aver_comp_2_sorted = aver_comp.sort_values(['z','y','x'], ascending = [True, True, True])
            aver_tens_2_sorted = aver_tens.sort_values(['z','y','x'], ascending = [True, True, True])


            delta_gamma = (aver_tens_2_sorted - aver_comp_2_sorted) 
            slip_values = abs(delta_gamma[['slip_' + str(i) for i in range(1,13)]]).to_numpy()
            
            max_slip_per_int = np.max(slip_values, axis = 1)
            max_slip_per_int_all_rolled.append(max_slip_per_int)
            

    ''' Plot rolled strained in the X direction '''

    # Define plot characteristics
    labelss = ['AR 1', 'AR 6', r'AR $\frac{1}{6}$', 'AR 1 - GS', 'AR 6 - GS', r'AR $\frac{1}{6}$ - GS']
    cfm     = ['r', 'b', 'g', 'r', 'b', 'g']
    linestyless = ['-', '-', '-', ':', ':', ':']
    
    list_x = [0,2,4,6,8,10]
    
    data_rolled_x = [max_slip_per_int_all_rolled[kk] for kk in list_x]

    n_bins = 100
    
    fig = plt.figure(facecolor="white", figsize=(6, 4), dpi=1200)
    for rr, valss in enumerate(data_rolled_x):
        
        n,x1,_ = plt.hist(valss, bins = n_bins, histtype = 'step', color = 'white')
        n1 = n.astype(int)
        bin_centers = 0.5*(x1[1:]+x1[:-1])
        plt.plot(bin_centers, n1, label = labelss[rr], color = cfm[rr], linestyle = linestyless[rr])    

    # plt.show()
    plt.legend(framealpha = 1, ncol = 2, loc = 'upper left')
    plt.xlim(max_slip_x_lims[0], max_slip_x_lims[1])
    plt.ylim(0, 4.4e4)
    plt.ylabel('Num')   
    plt.grid(True)
    plt.xlabel('Maximum plastic shear strain range')     
    plt.ticklabel_format(style = 'sci', axis='both', scilimits=(0,0), useMathText = True)    
    plt.tight_layout()
    plt.savefig( os.path.join(store_dirr, 'rolled_X_all_%d_bins_v3' % n_bins) )
    plt.close()   


    # Plot a zoomed-in figure at the highest plastic shear strain range values

    n_bins = 100
    
    fig = plt.figure(facecolor="white", figsize=(4, 2.666), dpi=1200)
    for rr, valss in enumerate(data_rolled_x):
    # for rr in plt_order:
        
        n,x1,_ = plt.hist(valss, bins = n_bins, histtype = 'step', color = 'white')
        n1 = n.astype(int)
        bin_centers = 0.5*(x1[1:]+x1[:-1])
        plt.plot(bin_centers, n1, label = labelss[rr], color = cfm[rr], linestyle = linestyless[rr])    

    # plt.show()
    plt.xlim(max_slip_x_lims[0], max_slip_x_lims[1])
    plt.xlim(3e-3, max_slip_x_lims[1])
    plt.ylim(0, 100)
    plt.grid(True)  
    plt.ticklabel_format(style = 'sci', axis='x', scilimits=(0,0), useMathText = True)    
    plt.tight_layout()
    plt.savefig( os.path.join(store_dirr, 'rolled_X_all_%d_bins_v3_ZOOM' % n_bins) )
    plt.close()   






    ''' Plot rolled strained in the Z direction '''

    list_z = [1,3,5,7,9,11]
    
    data_rolled_z = [max_slip_per_int_all_rolled[kk] for kk in list_z]
    
    n_bins = 100
    
    fig = plt.figure(facecolor="white", figsize=(6, 4), dpi=1200)
    for rr, valss in enumerate(data_rolled_z):
    # for rr in plt_order:
        
        n,x1,_ = plt.hist(valss, bins = n_bins, histtype = 'step', color = 'white')
        n1 = n.astype(int)
        bin_centers = 0.5*(x1[1:]+x1[:-1])
        plt.plot(bin_centers, n1, label = labelss[rr], color = cfm[rr], linestyle = linestyless[rr])    

    # plt.show()
    plt.xlim(max_slip_x_lims[0], max_slip_x_lims[1])

    plt.ylim(0, 4.4e4)
    plt.ylabel('Num')   
    plt.grid(True)
    plt.xlabel('Maximum plastic shear strain range')     
    plt.ticklabel_format(style = 'sci', axis='both', scilimits=(0,0), useMathText = True)    
    plt.tight_layout()
    plt.savefig( os.path.join(store_dirr, 'rolled_Z_all_%d_bins_v3' % n_bins) )
    plt.close()   


    # Plot a zoomed-in figure at the highest plastic shear strain range values

    n_bins = 100
    
    fig = plt.figure(facecolor="white", figsize=(4, 2.666), dpi=1200)
    for rr, valss in enumerate(data_rolled_z):
    # for rr in plt_order:
        
        n,x1,_ = plt.hist(valss, bins = n_bins, histtype = 'step', color = 'white')
        n1 = n.astype(int)
        bin_centers = 0.5*(x1[1:]+x1[:-1])
        plt.plot(bin_centers, n1, label = labelss[rr], color = cfm[rr], linestyle = linestyless[rr])    

    #plt.show()
    plt.legend(framealpha = 1, ncol = 2)
    plt.xlim(max_slip_x_lims[0], max_slip_x_lims[1])
    plt.xlim(3e-3, max_slip_x_lims[1])
    plt.ylim(0, 100)
    plt.grid(True)
    plt.ticklabel_format(style = 'sci', axis='x', scilimits=(0,0), useMathText = True)    
    plt.tight_layout()
    plt.savefig( os.path.join(store_dirr, 'rolled_Z_all_%d_bins_v3_ZOOM' % n_bins) )
    plt.close()   

def main():
    grain_size_paper_max_slip()


if __name__ == "__main__":
    main()