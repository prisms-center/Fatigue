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

def extract_stress_strain_curves_section_3():

    # Define directories
    directories = [os.path.join(DIR_LOC, r'Section_3\stress_strain\90^3_cubic_fully_periodic_equiaxed\_Random_equiaxed_0'),
                   os.path.join(DIR_LOC, r'Section_3\stress_strain\90^3_cubic_fully_periodic_elongated\_Random_equiaxed_0'),
                   os.path.join(DIR_LOC, r'Section_3\stress_strain\90^3_random_fully_periodic_equiaxed_additional_5\_Random_equiaxed_0'),
                   os.path.join(DIR_LOC, r'Section_3\stress_strain\90^3_random_fully_periodic_elongated\_Random_equiaxed_0'),
                   os.path.join(DIR_LOC, r'Section_3\stress_strain\90^3_rolled_fully_periodic_equiaxed\_Random_equiaxed_0'),
                   os.path.join(DIR_LOC, r'Section_3\stress_strain\90^3_rolled_fully_periodic_elongated\_Random_equiaxed_0'),
                   os.path.join(DIR_LOC, r'Section_3\stress_strain\160^3_cubic_fully_periodic_equiaxed\_Random_equiaxed_0'),
                   os.path.join(DIR_LOC, r'Section_3\stress_strain\160^3_cubic_fully_periodic_elongated\_Random_equiaxed_0'),
                   os.path.join(DIR_LOC, r'Section_3\stress_strain\160^3_random_fully_periodic_equiaxed_additional_5\_Random_equiaxed_0'),
                   os.path.join(DIR_LOC, r'Section_3\stress_strain\160^3_random_fully_periodic_elongated\_Random_equiaxed_0'),
                   os.path.join(DIR_LOC, r'Section_3\stress_strain\160^3_rolled_fully_periodic_equiaxed\_Random_equiaxed_0'),
                   os.path.join(DIR_LOC, r'Section_3\stress_strain\160^3_rolled_fully_periodic_elongated\_Random_equiaxed_0')]
                   
    # Define where to save plots
    store_dirr = os.path.join(DIR_LOC, r'plots')
    os.chdir(store_dirr)   
    
    # Initialize empty arrays
    all_strain_data = []
    all_stress_data = []

    for directory in directories:

        file_dirr = os.path.join(directory, 'stressstrain.txt')
        
        # Read in data using pandas module
        data2 = pd.read_csv(file_dirr, index_col = False, delimiter = "\t")

        # Get average of strain in Z 
        strain_z_temp = data2['Exx'].to_numpy()
        strain_z_temp = np.insert(strain_z_temp, 0, 0, axis=0)
        
        # Get average of stress in Z 
        stress_z_temp = data2['Txx'].to_numpy()
        stress_z_temp = np.insert(stress_z_temp, 0, 0, axis=0)
        
        all_strain_data.append(strain_z_temp)
        all_stress_data.append(stress_z_temp)


    
    # Plot stress-strain curves
    
    colorss = ['r', 'r', 'b', 'b', 'g', 'g']
    linestyless = ['-', ':', '-', ':', '-', ':',]
    labells = ['Cubic Equiaxed', 'Cubic Elongated', 'Random Equiaxed', 'Random Elongated', 'Rolled Equiaxed', 'Rolled Elongated']
    
    
    fig = plt.figure(facecolor="white", figsize=(5.5, 3.666), dpi=1200)    
    for kkk in np.arange(6):

        plt.plot(all_strain_data[kkk], all_stress_data[kkk], color = colorss[kkk], linestyle = linestyless[kkk], label = labells[kkk], linewidth = 1.5)

    plt.legend(framealpha = 1, fontsize = 8)
    
    # plt.show()
    plt.ylabel('Stress [MPa]')   
    plt.grid(True)
    plt.xlabel('Strain')
    plt.ticklabel_format(style = 'sci', axis='x', scilimits=(0,0), useMathText = True)        
    plt.tight_layout()
    plt.savefig(os.path.join(store_dirr, '7500_grain_textured_stress_strain'))
    plt.close()   
    


    colorss = ['r', 'r', 'b', 'b', 'g', 'g']
    linestyless = ['-', ':', '-', ':', '-', ':',]
    labells = ['Cubic Equiaxed', 'Cubic Elongated', 'Random Equiaxed', 'Random Elongated', 'Rolled Equiaxed', 'Rolled Elongated']
    
    
    fig = plt.figure(facecolor="white", figsize=(5.5, 3.666), dpi=1200)    
    for kkk in np.arange(6,12):

        plt.plot(all_strain_data[kkk], all_stress_data[kkk], color = colorss[kkk-6], linestyle = linestyless[kkk-6], label = labells[kkk-6], linewidth = 1.5)

    plt.legend(framealpha = 1, fontsize = 8)
    
    # plt.show()
    plt.ylabel('Stress [MPa]')   
    plt.grid(True)
    plt.xlabel('Strain')
    plt.ticklabel_format(style = 'sci', axis='x', scilimits=(0,0), useMathText = True)        
    plt.tight_layout()
    plt.savefig(os.path.join(store_dirr, '41000_grain_textured_stress_strain'))
    plt.close()       
  
def extract_stress_strain_curves_section_6():

    # Define directories
    directories = [os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\stress_strain\cropped_to_72_72_72_original_orientations\_Random_equiaxed_0'),
                   os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\stress_strain\Layer5_Largest_20Percent_Grain\_Random_equiaxed_0'),
                   os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\stress_strain\Layer5_Largest_20Percent_Grain\_Random_equiaxed_1'),
                   os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\stress_strain\Layer5_Largest_20Percent_Grain\_Random_equiaxed_2'),
                   os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\stress_strain\Layer5_Largest_20Percent_Grain\_Random_equiaxed_3'),
                   os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\stress_strain\Layer5_Largest_20Percent_Grain\_Random_equiaxed_4'),
                   os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\stress_strain\original_cropped_72^3_region_around_third_highest_FIP_grain\_Random_equiaxed_0'),
                   os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\stress_strain\Layer5_Largest_20Percent_Grain\_Random_equiaxed_0'),
                   os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\stress_strain\Layer5_Largest_20Percent_Grain\_Random_equiaxed_1'),
                   os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\stress_strain\Layer5_Largest_20Percent_Grain\_Random_equiaxed_2'),
                   os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\stress_strain\Layer5_Largest_20Percent_Grain\_Random_equiaxed_3'),
                   os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\stress_strain\Layer5_Largest_20Percent_Grain\_Random_equiaxed_4')]

    # Define where to save plots
    store_dirr = os.path.join(DIR_LOC, r'plots')
    os.chdir(store_dirr)    
    
    # Plot stress-strain curves
    all_strain_data = []
    all_stress_data = []

    for directory in directories:

        file_dirr = os.path.join(directory, 'stressstrain.txt')
        
        # Read in data using pandas module
        data2 = pd.read_csv(file_dirr, index_col = False, delimiter = "\t")

        # Get average of strain in Z 
        strain_z_temp = data2['Exx'].to_numpy()
        strain_z_temp = np.insert(strain_z_temp, 0, 0, axis=0)
        
        
        # Get average of stress in Z 
        stress_z_temp = data2['Txx'].to_numpy()
        stress_z_temp = np.insert(stress_z_temp, 0, 0, axis=0)
        
        all_strain_data.append(strain_z_temp)
        all_stress_data.append(stress_z_temp)


    
    # Plot stress-strain curve    
    
    colorss = ['r', 'm', 'b', 'c', 'g', 'k']
    linestyless = [':', '-.', '--', '-', ':', '-.']
    labells = ['Original', 'Alt 1', 'Alt 2', 'Alt 3', 'Alt 4', 'Alt 5']
    
    
    fig = plt.figure(facecolor="white", figsize=(5.5, 3.666), dpi=1200)    
    for kkk in np.arange(6):
    
        plt.plot(all_strain_data[kkk], all_stress_data[kkk], color = colorss[kkk], marker = 'o', linestyle = linestyless[kkk], label = labells[kkk], linewidth = 1.5, markersize = 0.25)

    plt.legend(framealpha = 1, fontsize = 8)
    
    # plt.show()
    plt.ylabel('Stress [MPa]')   
    plt.grid(True)
    plt.xlabel('Strain')
    plt.ticklabel_format(style = 'sci', axis='x', scilimits=(0,0), useMathText = True)        
    plt.tight_layout()
    plt.savefig(os.path.join(store_dirr, '1st_highest_GOI_stress_strain_variation'))
    plt.close()   
    
    

    colorss = ['r', 'm', 'b', 'c', 'g', 'k']
    linestyless = [':', '-.', '--', '-', ':', '-.']
    labells = ['Original', 'Alt 1', 'Alt 2', 'Alt 3', 'Alt 4', 'Alt 5']
    
    fig = plt.figure(facecolor="white", figsize=(5.5, 3.666), dpi=1200)    
    for kkk in np.arange(6,12):
    
        plt.plot(all_strain_data[kkk], all_stress_data[kkk], color = colorss[kkk-6], linestyle = linestyless[kkk-6], label = labells[kkk-6], linewidth = 1.5)

    plt.legend(framealpha = 1, fontsize = 8)
    
    # plt.show()
    plt.ylabel('Stress [MPa]')   
    plt.grid(True)
    plt.xlabel('Strain')
    plt.ticklabel_format(style = 'sci', axis='x', scilimits=(0,0), useMathText = True)        
    plt.tight_layout()
    plt.savefig(os.path.join(store_dirr, '3rd_highest_GOI_stress_strain_variation'))
    plt.close()       
    
  
def main():
    # Plot stress-strain curves for section 3
    extract_stress_strain_curves_section_3()
    
    # Plot stress-strain curves for section 6
    extract_stress_strain_curves_section_6()    

if __name__ == "__main__":
    main()




