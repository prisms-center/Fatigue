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

# Get name of directory that contains the PRISMS-Fatigue scripts
DIR_LOC = os.path.dirname(os.path.abspath(__file__))

# Define where to store plots
store_dirr = os.path.join(DIR_LOC, r'plots')

''' This script plots the convergence of FIPs and the cumulative effective plastic strain '''

def plot_FIP_convergence_for_five_cycles():
    ''' This function will plot the convergence of FIPs for a microstructure strained up to five fully-reversed cycles '''
    
    # Define directory
    directory = os.path.join(DIR_LOC, r'Section_3\7500_grain\7500_grain_random_equiaxed\extra_sims')
    
    # Define file name of FIPs
    fip_file_name = 'sub_band_averaged_highest_per_grain_FS_FIP_%d.csv'
    
    # Initialize arrays
    all_fips = []
    
    # Iterate through the four files with FIPs from the 2nd, 3rd, 4th, and 5th fully reversed cycles
    for num in range(4):
    
        # Get file name
        fname = os.path.join(directory, fip_file_name % (num + 2))

        # Read data
        data2 = pd.read_csv(fname, index_col = False)
        fips_temp = data2['SubBandAveragedFipValue']
        fips = fips_temp.to_numpy()
        
        # Store all fips in one array
        all_fips.append(fips)

    # Define plot labels and markers
    markerss = ['o', 's', '^', 'D']
    labelss = ['2 cycles', '3 cycles', '4 cycles', '5 cycles']
    
    # Plot data
    fig = plt.figure(facecolor="white", figsize=(6, 4), dpi=1200)

    for xx in range(4):
        plt.plot(range(1, len(all_fips[xx])+1), all_fips[xx], marker = markerss[xx], markersize = 2, linestyle = ':', label = labelss[xx])
        
    plt.legend(loc='upper right', framealpha = 1)# , fontsize = 8.5)
    
    # plt.show()
    plt.ylabel('Sub-band volume-averaged FIP')   
    plt.grid(True)
    plt.xlabel('Grain ID by FIP rank')
    plt.xscale('log')
    plt.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0), useMathText = True)
    plt.tight_layout()
    plt.savefig(os.path.join(store_dirr, 'FIP_convergence'))
    plt.close()        
    
    
    
    ''' This code will plot only the top 'zoom_fips' number of highest FIPs '''
    zoom_fips = 500
    
    # Plot data 
    fig = plt.figure(facecolor="white", figsize=(6, 4), dpi=1200)

    for xx in range(4):
        plt.plot(range(1, len(all_fips[xx][:zoom_fips])+1), all_fips[xx][:zoom_fips], marker = markerss[xx], markersize = 2, linestyle = ':', label = labelss[xx])

    plt.legend(loc='upper right', framealpha = 1)# , fontsize = 8.5)
    
    # plt.show()
    plt.ylabel('Sub-band volume-averaged FIP')   
    plt.grid(True)
    plt.xlabel('Grain ID by FIP rank')
    plt.xscale('log')
    plt.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0), useMathText = True)
    plt.tight_layout()
    plt.savefig(os.path.join(store_dirr, 'FIP_convergence_zoom_%d_fips' % zoom_fips))
    plt.close()        

def plot_section_3_microscale_response():
    
    # Define directories with files
    directories = [os.path.join(DIR_LOC, r'Section_3\7500_grain\7500_grain_cubic_elongated\extra_sims'),
                   os.path.join(DIR_LOC, r'Section_3\7500_grain\7500_grain_cubic_equiaxed\extra_sims'),
                   os.path.join(DIR_LOC, r'Section_3\7500_grain\7500_grain_random_elongated\extra_sims'),
                   os.path.join(DIR_LOC, r'Section_3\7500_grain\7500_grain_random_equiaxed\extra_sims'),
                   os.path.join(DIR_LOC, r'Section_3\7500_grain\7500_grain_rolled_elongated\extra_sims'),
                   os.path.join(DIR_LOC, r'Section_3\7500_grain\7500_grain_rolled_equiaxed\extra_sims'),
                   os.path.join(DIR_LOC, r'Section_3\41000_grain\41000_grain_cubic_elongated\extra_sims'),
                   os.path.join(DIR_LOC, r'Section_3\41000_grain\41000_grain_cubic_equiaxed\extra_sims'),
                   os.path.join(DIR_LOC, r'Section_3\41000_grain\41000_grain_random_elongated\extra_sims'),
                   os.path.join(DIR_LOC, r'Section_3\41000_grain\41000_grain_random_equiaxed\extra_sims'),
                   os.path.join(DIR_LOC, r'Section_3\41000_grain\41000_grain_rolled_elongated\extra_sims'),
                   os.path.join(DIR_LOC, r'Section_3\41000_grain\41000_grain_rolled_equiaxed\extra_sims')]
  
    
    # Check to see if compiled data already exists
    fname_EpEff_temp = 'Ep_eff_Section_3.p'
    fname_EpEff = os.path.join(store_dirr, fname_EpEff_temp)
    
    # If data has not been compiled, read in data and store as .p file
    if not os.path.exists(fname_EpEff):
        
        print('Reading EpEff from .csv')
        
        # Initialize list of data
        all_EpEff_data = []
        
        for directory in directories:
            # Specify files with simulations values at points of maximum tension (final point in the simulation)
            dirr_max_tens = os.path.join(directory, 'maxTen2.csv')
            
            print('Reading file %s' % dirr_max_tens)
            
            # Read in data using pandas module
            aver_tens = pd.read_csv(dirr_max_tens, index_col = False)
            # "index_col = False" means that the first column is NOT the index, which is the case in the quadrature output files here.
            
            # NOTE: the default PRISMS-Fatigue quadrature output columns correspond to the following values for each quadrature point (or element in the case of simulations with reduced integration elements)
            # The first four columns do not change during the simulation. The remaining columns correspond to current values of state variables
            # Grain ID, x position, y position, z position, plastic shear strain for slip systems 1 thru 12, stress normal to slip planes of slip systems 1 thru 12, plastic strain tensor in global directions (i.e., Ep11, Ep12, Ep13	Ep21, Ep22, Ep23, Ep31, Ep32, Ep33), Effective plastic strain

            aver_tens_2_sorted = aver_tens.sort_values(['z','y','x'], ascending = [True, True, True])
            
            eff_slip = aver_tens_2_sorted['EpEff'].to_numpy()
            
            # slip_values_np = slip_values.to_numpy()
            all_EpEff_data.append(eff_slip)

        h1 = open(fname_EpEff, 'wb')
        p.dump(all_EpEff_data,h1)
        h1.close()
        
    else:
        # Otherwise, just read in the data (this saves time while plotting...)
        
        print('Reading EpEff from .p')
        
        h1 = open(fname_EpEff, 'rb')
        all_EpEff_data = p.load(h1)
        h1.close()
    

    ''' Plot histograms '''
    
    
    ''' Plot equiaxed microstructures '''
    
    # Define legend labels, colors, and line styles
    labelss = ['~7,500    grain cubic', '~41,000  grain cubic', '~7,500    grain random', '~41,000  grain random', '~7,500    grain rolled', '~41,000  grain rolled']
    colorss = ['r', 'r', 'b', 'b', 'g', 'g']
    linestyless = [':', '-', ':', '-', ':', '-']
    
    # Specify the number of bins for the histograms
    n_bins = 100
 
    # Plot figure
    fig = plt.figure(facecolor="white", figsize=(6, 4), dpi=1200)
    
    # Specify the order in which samples should be plotted
    # This MUST match between the labels and directories specified above!
    manual_order_of_samples = [1, 7, 3, 9, 5, 11]
    
    for pp, ii in enumerate(manual_order_of_samples):
    
        n,x1,_ = plt.hist(all_EpEff_data[ii], bins = n_bins, histtype = 'step', color = 'white')
        n1 = n.astype(int)
        bin_centers = 0.5*(x1[1:]+x1[:-1])
        plt.plot(bin_centers, n1/len(all_EpEff_data[ii]), label = labelss[pp] , color = colorss[pp], linestyle = linestyless[pp])
    
    # plt.show()
    # plt.legend(bbox_to_anchor=(.6, 1.05), framealpha = 1, fontsize = 9)
    plt.legend(loc='upper left', framealpha = 1, fontsize = 9)
    plt.ylabel('Fraction')   
    plt.ylim(0,0.085)
    plt.xlim(0,1.e-2)
    plt.grid(True)
    plt.xlabel('Cumulative effective plastic strain')       
    plt.ticklabel_format(style = 'sci', axis='both', scilimits=(0,0), useMathText = True)    
    plt.tight_layout()
    plt.savefig(os.path.join(store_dirr, 'Section_3_microscale_EpEff_equiaxed'))
    plt.close()       


    
    
    ''' Plot elongated microstructures '''
    # Define legend labels, colors, and line styles
    labelss = ['~7,500    grain cubic', '~41,000  grain cubic', '~7,500    grain random', '~41,000  grain random', '~7,500    grain rolled', '~41,000  grain rolled']
    colorss = ['r', 'r', 'b', 'b', 'g', 'g']
    linestyless = [':', '-', ':', '-', ':', '-']
    
    # Specify the number of bins for the histograms
    n_bins = 100
    
    # Plot figure
    fig = plt.figure(facecolor="white", figsize=(6, 4), dpi=1200)
    
    # Specify the order in which samples should be plotted
    # This MUST match between the labels and directories specified above!
    manual_order_of_samples = [0, 6, 2, 8, 4, 10]
    
    for pp, ii in enumerate(manual_order_of_samples):
    
        n,x1,_ = plt.hist(all_EpEff_data[ii], bins = n_bins, histtype = 'step', color = 'white')
        n1 = n.astype(int)
        bin_centers = 0.5*(x1[1:]+x1[:-1])
        plt.plot(bin_centers, n1/len(all_EpEff_data[ii]), label = labelss[pp] , color = colorss[pp], linestyle = linestyless[pp])
    
    # plt.show()
    # plt.legend(bbox_to_anchor=(0.13, 1.15), loc='upper left', framealpha = 1, fontsize = 8)
    plt.legend(loc='upper left', framealpha = 1, fontsize = 9)
    plt.ylabel('Fraction')   
    plt.ylim(0,0.085)
    plt.xlim(0,1e-2)
    plt.grid(True)
    plt.xlabel('Cumulative effective plastic strain')       
    plt.ticklabel_format(style = 'sci', axis='both', scilimits=(0,0), useMathText = True)    
    plt.tight_layout()
    plt.savefig(os.path.join(store_dirr, 'Section_3_microscale_EpEff_elongated'))
    plt.close()       

def main():
    # Plot the convergence of FIPs as a function of number of straining cycles
    plot_FIP_convergence_for_five_cycles()
    
    # Plot the distributions of cumulative effective plastic strain 
    plot_section_3_microscale_response()

if __name__ == "__main__":
    main()