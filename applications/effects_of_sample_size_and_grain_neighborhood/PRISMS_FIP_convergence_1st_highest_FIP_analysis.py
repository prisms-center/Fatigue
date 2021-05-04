import os
import numpy as np
import scipy.stats as ss
import sklearn.metrics as sm
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as plt_cols
import matplotlib.cm as cm
import scipy.ndimage.filters as filters
import pandas as pd
import matplotlib.ticker as ticker
import operator
import glob
import pickle as p
import scipy
import re
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple

# Get name of directory that contains the PRISMS-Fatigue scripts, i.e., this script
DIR_LOC = os.path.dirname(os.path.abspath(__file__))

# Define grain ID of the grain with the third highest FIP, indexed at 0.
# This grain's information is therefore 556 in the FeatureData_FakeMatl_0.csv file
grain_num_of_interest = 2423

# Define where to store plots
store_dirr = os.path.join(DIR_LOC, r'plots')

# Define templates for FIP pickle files for the FS_FIP, normal stress, and plastic shear strain range
file_name_template_master = ['sub_band_averaged_FS_FIP_pickle_%d.p', 'sub_band_averaged_normal_stress_pickle_%d.p', 'sub_band_averaged_plastic_shear_strain_range_pickle_%d.p']

def read_FIPs_from_certain_grain(directory, file_name_template, num_inst):
    # Additional unused function to read in the largest sub-band averaged FIPs (one per grain) from a single microstructure instantiation
    # This is particularly useful when considering a very large SVE with more than ~10,000 grains as this function can take a long time!
    # Go to directory
    
    tmp_dir = os.getcwd()
    os.chdir(directory)
    
    # Specify name of pickle file with sub-band averaged FIPs
    fname = file_name_template % num_inst    
    

    
    # Initialize list of just FIPs
    new_all_fs_fips = []

    # Initialize list to keep track of which grains have already been considered
    added_g = []
    
    # Read in FIPs
    h1 = open(fname,'rb')
    fips = p.load(h1)
    h1.close()
    
    # Sort in descending order
    sorted_fips = sorted(fips.items(), key=operator.itemgetter(1))
    sorted_fips.reverse()
    
    # Specify how many of the highest FIPs per grain should be imported. Typically, only the few hundred highest FIPs are of interest
    # This significantly speeds up this algorithm!
    # IMPORTANT: If this is set below the number of grains in the instantiation, the function will fail! 
    get_num_FIPs = len(sorted_fips)
    
    # Initialize array with more detailed FIP data
    # FIP, grain number, slip system number, layer number, sub-band region number
    all_data = np.zeros((get_num_FIPs,5))
    
    # Main counter
    nn = 0
    
    # Track counter
    mm = 0    

    while len(added_g) < get_num_FIPs:
    
        if sorted_fips[nn][0][0] not in added_g:
            added_g.append(sorted_fips[nn][0][0])
            
            all_data[mm][0] = sorted_fips[nn][1]
            all_data[mm][1] = sorted_fips[nn][0][0]
            all_data[mm][2] = sorted_fips[nn][0][1]
            all_data[mm][3] = sorted_fips[nn][0][2]
            all_data[mm][4] = sorted_fips[nn][0][3]
            mm += 1
            new_all_fs_fips.append(sorted_fips[nn][1])
            # print(mm)
        
        if sorted_fips[nn][0][0] == grain_num_of_interest:
            break
            
        nn += 1     
    
    os.chdir(tmp_dir)
    # return new_all_fs_fips
    return new_all_fs_fips, all_data[0:mm], added_g 

def read_normal_stress_for_COV():
    
    # This function reads in the maximum normal stress in the grain of interest to calculate/plot coefficients of variation
  
    file_name_template = file_name_template_master[1]
  
    # 10
    ''' Change largest NN in first layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\cropped_72_72_72_change_single_largest_neighbor_volume_orientation')
    num_inst = 10

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_first_NN_new_FIPs = []


    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        grain_2424_change_first_NN_new_FIPs.append(total_all_data[ii].transpose()[0][kk])

    new_largest_first_NN_new_FIPs = []
    for tt in total_all_data:
        new_largest_first_NN_new_FIPs.append(tt[0][0])

    grain_with_new_largest_FIP_1st_NN = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_1st_NN.append(tt[0])

    # 5
    ''' Change largest NN in second layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\Change_single_largest_grain_in_2nd_layer')
    num_inst = 5

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_second_NN_new_FIPs = []
    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        
        grain_2424_change_second_NN_new_FIPs.append(total_all_data[ii].transpose()[0][kk])

    new_largest_second_NN_new_FIPs = []
    for tt in total_all_data:
        new_largest_second_NN_new_FIPs.append(tt[0][0])


    grain_with_new_largest_FIP_2nd_NN = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_2nd_NN.append(tt[0])

    # 5
    ''' Change largest NN in third layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\Change_single_largest_grain_in_3rd_layer')
    num_inst = 5

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_third_NN_new_FIPs = []
    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        
        grain_2424_change_third_NN_new_FIPs.append(total_all_data[ii].transpose()[0][kk])

    new_largest_third_NN_new_FIPs = []
    for tt in total_all_data:
        new_largest_third_NN_new_FIPs.append(tt[0][0])


    grain_with_new_largest_FIP_3rd_NN = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_3rd_NN.append(tt[0])

    # 5
    ''' Change largest NN in fourth layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\Change_single_largest_grain_in_4th_layer')
    num_inst = 5

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_fourth_NN_new_FIPs = []
    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        
        grain_2424_change_fourth_NN_new_FIPs.append(total_all_data[ii].transpose()[0][kk])

    new_largest_fourth_NN_new_FIPs = []
    for tt in total_all_data:
        new_largest_fourth_NN_new_FIPs.append(tt[0][0])


    grain_with_new_largest_FIP_4th_NN = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_4th_NN.append(tt[0])

    # 5
    ''' Change largest NN in fifth layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\Change_single_largest_grain_in_5th_layer')
    num_inst = 5

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_fifth_NN_new_FIPs = []
    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        
        grain_2424_change_fifth_NN_new_FIPs.append(total_all_data[ii].transpose()[0][kk])

    new_largest_fifth_NN_new_FIPs = []
    for tt in total_all_data:
        new_largest_fifth_NN_new_FIPs.append(tt[0][0])


    grain_with_new_largest_FIP_5th_NN = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_5th_NN.append(tt[0])


    # All the above still take place in grain 2424 (indexed at 1)











    # 5
    ''' Change ALL NN in first layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\cropped_72_72_72_different_first_neighbor_orientations')
    num_inst = 5

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_ALL_first_NN_new_FIPs = []


    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        grain_2424_change_ALL_first_NN_new_FIPs.append(total_all_data[ii].transpose()[0][kk])

    new_largest_ALL_first_NN_new_FIPs = []
    for tt in total_all_data:
        new_largest_ALL_first_NN_new_FIPs.append(tt[0][0])

    grain_with_new_largest_FIP_ALL_1st_NN = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_ALL_1st_NN.append(tt[0])

    # 5
    ''' Change ALL NN in second layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\cropped_72_72_72_different_second_neighbor_orientations')
    num_inst = 5

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_ALL_second_NN_new_FIPs = []
    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        
        grain_2424_change_ALL_second_NN_new_FIPs.append(total_all_data[ii].transpose()[0][kk])

    new_largest_ALL_second_NN_new_FIPs = []
    for tt in total_all_data:
        new_largest_ALL_second_NN_new_FIPs.append(tt[0][0])


    grain_with_new_largest_FIP_ALL_2nd_NN = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_ALL_2nd_NN.append(tt[0])

    # 5
    ''' Change ALL NN in third layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\cropped_72_72_72_different_3rd_neighbor_orientations')
    num_inst = 5

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_ALL_third_NN_new_FIPs = []
    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        
        grain_2424_change_ALL_third_NN_new_FIPs.append(total_all_data[ii].transpose()[0][kk])

    new_largest_ALL_third_NN_new_FIPs = []
    for tt in total_all_data:
        new_largest_ALL_third_NN_new_FIPs.append(tt[0][0])


    grain_with_new_largest_FIP_ALL_3rd_NN = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_ALL_3rd_NN.append(tt[0])

    # 5
    ''' Change ALL NN in fourth layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\cropped_72_72_72_different_4th_neighbor_orientations')
    num_inst = 5

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_ALL_fourth_NN_new_FIPs = []
    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        
        grain_2424_change_ALL_fourth_NN_new_FIPs.append(total_all_data[ii].transpose()[0][kk])

    new_largest_ALL_fourth_NN_new_FIPs = []
    for tt in total_all_data:
        new_largest_ALL_fourth_NN_new_FIPs.append(tt[0][0])


    grain_with_new_largest_FIP_ALL_4th_NN = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_ALL_4th_NN.append(tt[0])

    # 5
    ''' Change ALL NN in fifth layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\cropped_72_72_72_different_5th_neighbor_orientations')
    num_inst = 5

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_ALL_fifth_NN_new_FIPs = []
    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        
        grain_2424_change_ALL_fifth_NN_new_FIPs.append(total_all_data[ii].transpose()[0][kk])

    new_largest_ALL_fifth_NN_new_FIPs = []
    for tt in total_all_data:
        new_largest_ALL_fifth_NN_new_FIPs.append(tt[0][0])


    grain_with_new_largest_FIP_ALL_5th_NN = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_ALL_5th_NN.append(tt[0])





    # 10
    ''' Change NN with most shared area in first layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\cropped_72_72_72_change_single_largest_neighbor_shared_area_orientation')
    num_inst = 10

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_first_NN_most_shared_area = []
    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        
        grain_2424_change_first_NN_most_shared_area.append(total_all_data[ii].transpose()[0][kk])

    new_largest_first_NN_most_shared_area_new_FIPs = []
    for tt in total_all_data:
        new_largest_first_NN_most_shared_area_new_FIPs.append(tt[0][0])


    grain_with_new_largest_FIP_1st_NN_most_shared_area = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_1st_NN_most_shared_area.append(tt[0])


    # 1
    ''' COMPARE TO ORIGINAL FROM 250^3 '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\cropped_to_72_72_72_original_orientations')
    num_inst = 1

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_original = []
    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        
        grain_2424_original.append(total_all_data[ii].transpose()[0][kk])

    new_original_FIPs = []
    for tt in total_all_data:
        new_original_FIPs.append(tt[0][0])


    oringial_FIP_same_orientation = []
    for tt in total_added_g:
        oringial_FIP_same_orientation.append(tt[0])


    # All cases above have largest FIP in grain 2424.



    # NOTE: Highest FIPs when changing a SINGLE nearest neighbor in all five layers remains in grain 2424
    #       Highest FIPs when changing ALL nearest neighbors in first and second layer is now 1567, whereas it is 2424 when changing all third, fourth, and fifth NN orientations!
    #       The last five all occur in 2424!


    # https://towardsdatascience.com/scattered-boxplots-graphing-experimental-results-with-matplotlib-seaborn-and-pandas-81f9fa8a1801

    dataset1 = [grain_2424_change_first_NN_most_shared_area, 
                grain_2424_change_first_NN_new_FIPs, grain_2424_change_second_NN_new_FIPs, grain_2424_change_third_NN_new_FIPs, grain_2424_change_fourth_NN_new_FIPs, grain_2424_change_fifth_NN_new_FIPs, 
                grain_2424_change_ALL_first_NN_new_FIPs, grain_2424_change_ALL_second_NN_new_FIPs, grain_2424_change_ALL_third_NN_new_FIPs, grain_2424_change_ALL_fourth_NN_new_FIPs, grain_2424_change_ALL_fifth_NN_new_FIPs, grain_2424_original]


    fname = '1st_highest_normal_stress_variation_first_plot_data.p'
    h1 = open(os.path.join(store_dirr, fname), 'wb')
    p.dump(dataset1, h1)
    h1.close()
    
    return dataset1
   
def read_and_plot_main_FIP_PSSR(file_name_template, type_plot):
    # This function will read in the FIPs to create the first plot for the grain that manifests the 1st highest FIP in the 160,000 grain microstructure
  
    # 10
    ''' Change largest NN in first layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\cropped_72_72_72_change_single_largest_neighbor_volume_orientation')
    num_inst = 10

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_first_NN_new_FIPs = []


    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        grain_2424_change_first_NN_new_FIPs.append(total_all_data[ii].transpose()[0][kk])

    new_largest_first_NN_new_FIPs = []
    for tt in total_all_data:
        new_largest_first_NN_new_FIPs.append(tt[0][0])

    grain_with_new_largest_FIP_1st_NN = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_1st_NN.append(tt[0])

    # 5
    ''' Change largest NN in second layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\Change_single_largest_grain_in_2nd_layer')
    num_inst = 5

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_second_NN_new_FIPs = []
    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        
        grain_2424_change_second_NN_new_FIPs.append(total_all_data[ii].transpose()[0][kk])

    new_largest_second_NN_new_FIPs = []
    for tt in total_all_data:
        new_largest_second_NN_new_FIPs.append(tt[0][0])


    grain_with_new_largest_FIP_2nd_NN = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_2nd_NN.append(tt[0])

    # 5
    ''' Change largest NN in third layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\Change_single_largest_grain_in_3rd_layer')
    num_inst = 5

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_third_NN_new_FIPs = []
    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        
        grain_2424_change_third_NN_new_FIPs.append(total_all_data[ii].transpose()[0][kk])

    new_largest_third_NN_new_FIPs = []
    for tt in total_all_data:
        new_largest_third_NN_new_FIPs.append(tt[0][0])


    grain_with_new_largest_FIP_3rd_NN = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_3rd_NN.append(tt[0])

    # 5
    ''' Change largest NN in fourth layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\Change_single_largest_grain_in_4th_layer')
    num_inst = 5

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_fourth_NN_new_FIPs = []
    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        
        grain_2424_change_fourth_NN_new_FIPs.append(total_all_data[ii].transpose()[0][kk])

    new_largest_fourth_NN_new_FIPs = []
    for tt in total_all_data:
        new_largest_fourth_NN_new_FIPs.append(tt[0][0])


    grain_with_new_largest_FIP_4th_NN = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_4th_NN.append(tt[0])

    # 5
    ''' Change largest NN in fifth layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\Change_single_largest_grain_in_5th_layer')
    num_inst = 5

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_fifth_NN_new_FIPs = []
    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        
        grain_2424_change_fifth_NN_new_FIPs.append(total_all_data[ii].transpose()[0][kk])

    new_largest_fifth_NN_new_FIPs = []
    for tt in total_all_data:
        new_largest_fifth_NN_new_FIPs.append(tt[0][0])


    grain_with_new_largest_FIP_5th_NN = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_5th_NN.append(tt[0])


    # All the above still take place in grain 2424 (indexed at 1)











    # 5
    ''' Change ALL NN in first layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\cropped_72_72_72_different_first_neighbor_orientations')
    num_inst = 5

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_ALL_first_NN_new_FIPs = []


    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        grain_2424_change_ALL_first_NN_new_FIPs.append(total_all_data[ii].transpose()[0][kk])

    new_largest_ALL_first_NN_new_FIPs = []
    for tt in total_all_data:
        new_largest_ALL_first_NN_new_FIPs.append(tt[0][0])

    grain_with_new_largest_FIP_ALL_1st_NN = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_ALL_1st_NN.append(tt[0])

    # 5
    ''' Change ALL NN in second layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\cropped_72_72_72_different_second_neighbor_orientations')
    num_inst = 5

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_ALL_second_NN_new_FIPs = []
    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        
        grain_2424_change_ALL_second_NN_new_FIPs.append(total_all_data[ii].transpose()[0][kk])

    new_largest_ALL_second_NN_new_FIPs = []
    for tt in total_all_data:
        new_largest_ALL_second_NN_new_FIPs.append(tt[0][0])


    grain_with_new_largest_FIP_ALL_2nd_NN = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_ALL_2nd_NN.append(tt[0])

    # 5
    ''' Change ALL NN in third layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\cropped_72_72_72_different_3rd_neighbor_orientations')
    num_inst = 5

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_ALL_third_NN_new_FIPs = []
    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        
        grain_2424_change_ALL_third_NN_new_FIPs.append(total_all_data[ii].transpose()[0][kk])

    new_largest_ALL_third_NN_new_FIPs = []
    for tt in total_all_data:
        new_largest_ALL_third_NN_new_FIPs.append(tt[0][0])


    grain_with_new_largest_FIP_ALL_3rd_NN = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_ALL_3rd_NN.append(tt[0])

    # 5
    ''' Change ALL NN in fourth layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\cropped_72_72_72_different_4th_neighbor_orientations')
    num_inst = 5

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_ALL_fourth_NN_new_FIPs = []
    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        
        grain_2424_change_ALL_fourth_NN_new_FIPs.append(total_all_data[ii].transpose()[0][kk])

    new_largest_ALL_fourth_NN_new_FIPs = []
    for tt in total_all_data:
        new_largest_ALL_fourth_NN_new_FIPs.append(tt[0][0])


    grain_with_new_largest_FIP_ALL_4th_NN = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_ALL_4th_NN.append(tt[0])

    # 5
    ''' Change ALL NN in fifth layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\cropped_72_72_72_different_5th_neighbor_orientations')
    num_inst = 5

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_ALL_fifth_NN_new_FIPs = []
    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        
        grain_2424_change_ALL_fifth_NN_new_FIPs.append(total_all_data[ii].transpose()[0][kk])

    new_largest_ALL_fifth_NN_new_FIPs = []
    for tt in total_all_data:
        new_largest_ALL_fifth_NN_new_FIPs.append(tt[0][0])


    grain_with_new_largest_FIP_ALL_5th_NN = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_ALL_5th_NN.append(tt[0])


    # The first two cases above have highest FS_FIP in grain number 1566 (indexed at 1); the last three are back at 2424.

    # The first two cases above have highest PSSR in grain number 3968 (indexed at 1); the last three are back at 2424.















    # 10
    ''' Change NN with most shared area in first layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\cropped_72_72_72_change_single_largest_neighbor_shared_area_orientation')
    num_inst = 10

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_first_NN_most_shared_area = []
    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        
        grain_2424_change_first_NN_most_shared_area.append(total_all_data[ii].transpose()[0][kk])

    new_largest_first_NN_most_shared_area_new_FIPs = []
    for tt in total_all_data:
        new_largest_first_NN_most_shared_area_new_FIPs.append(tt[0][0])


    grain_with_new_largest_FIP_1st_NN_most_shared_area = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_1st_NN_most_shared_area.append(tt[0])


    # 1
    ''' Largest NN has same orientation '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\cropped_72_72_72_largest_neighbor_has_same_orientation')
    num_inst = 1

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_largest_NN_same_orientation = []
    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        
        grain_2424_largest_NN_same_orientation.append(total_all_data[ii].transpose()[0][kk])

    new_largest_NN_same_orientation_new_FIPs = []
    for tt in total_all_data:
        new_largest_NN_same_orientation_new_FIPs.append(tt[0][0])


    grain_with_new_largest_FIP_NN_same_orientation = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_NN_same_orientation.append(tt[0])

    # 1
    ''' NN with most shared surface area has same orientation '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\cropped_72_72_72_neighbor_with_most_shared_area_has_same_orientation')
    num_inst = 1

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_most_shared_area_NN_same_orientation = []
    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        
        grain_2424_most_shared_area_NN_same_orientation.append(total_all_data[ii].transpose()[0][kk])

    new_most_shared_area_NN_same_orientation_new_FIPs = []
    for tt in total_all_data:
        new_most_shared_area_NN_same_orientation_new_FIPs.append(tt[0][0])


    grain_with_new_most_shared_area_FIP_NN_same_orientation = []
    for tt in total_added_g:
        grain_with_new_most_shared_area_FIP_NN_same_orientation.append(tt[0])

    # 1
    ''' COMPARE TO ORIGINAL FROM 250^3 '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\cropped_to_72_72_72_original_orientations')
    num_inst = 1

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_original = []
    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        
        grain_2424_original.append(total_all_data[ii].transpose()[0][kk])

    new_original_FIPs = []
    for tt in total_all_data:
        new_original_FIPs.append(tt[0][0])


    oringial_FIP_same_orientation = []
    for tt in total_added_g:
        oringial_FIP_same_orientation.append(tt[0])


    # All cases above have largest FIP in grain 2424.



    # NOTE: Highest FIPs when changing a SINGLE nearest neighbor in all five layers remains in grain 2424
    #       Highest FIPs when changing ALL nearest neighbors in first and second layer is now 1567, whereas it is 2424 when changing all third, fourth, and fifth NN orientations!
    #       The last five all occur in 2424!


    # https://towardsdatascience.com/scattered-boxplots-graphing-experimental-results-with-matplotlib-seaborn-and-pandas-81f9fa8a1801

    dataset1 = [grain_2424_change_first_NN_most_shared_area, 
                new_largest_first_NN_new_FIPs, new_largest_second_NN_new_FIPs, new_largest_third_NN_new_FIPs, new_largest_fourth_NN_new_FIPs, new_largest_fifth_NN_new_FIPs, 
                grain_2424_change_ALL_first_NN_new_FIPs, grain_2424_change_ALL_second_NN_new_FIPs, new_largest_ALL_third_NN_new_FIPs, new_largest_ALL_fourth_NN_new_FIPs, new_largest_ALL_fifth_NN_new_FIPs,
                grain_2424_largest_NN_same_orientation, grain_2424_most_shared_area_NN_same_orientation]


    names1 = ['$1^{st}$ NN with most SA', 
              'Largest $1^{st}$ NN','Largest $2^{nd}$ NN', 'Largest $3^{rd}$ NN', 'Largest $4^{th}$ NN', 'Largest $5^{th}$ NN', 
              'All 13 $1^{st}$ NNs','All 68 $2^{nd}$ NNs', 'All 159 $3^{rd}$ NNs', 'All 303 $4^{th}$ NNs', 'All 517 $5^{th}$ NNs',
              'Largest $1^{st}$ NN, same ori', '$1^{st}$ NN with most SA, same ori']

    vals, names, xs = [],[],[]

    for i, col in enumerate(names1):
        vals.append(dataset1[i])
        names.append(col)
        xs.append(np.random.normal(i + 1, 0.04, len(dataset1[i])))  # adds jitter to the data points - can be adjusted


    dataset2 = [new_largest_ALL_first_NN_new_FIPs, new_largest_ALL_second_NN_new_FIPs]
    vals_1, xs_1 = [], []

    for kk in range(2):
        vals_1.append(dataset2[kk])
        xs_1.append(np.random.normal(kk + 7, 0.04, len(dataset2[kk])))  # adds jitter to the data points - can be adjusted
        
    # Store plotting data in pickle file!

    if type_plot == 'FS_FIP':
        fname = '1st_highest_FS_FIP_variation_first_plot_data.p'
    elif type_plot == 'PSSR':
        fname = '1st_highest_PSSR_variation_first_plot_data.p'
    h1 = open(os.path.join(store_dirr, fname), 'wb')
    p.dump([dataset1, dataset2, names1], h1)
    h1.close()

    fontsizee = 7

    fig = plt.figure(facecolor="white", figsize=(7.5, 5), dpi=1200)
    plt.rcParams["mathtext.default"] = "regular"
    plt.boxplot(vals[0:11]+[1,1], labels=names, zorder = 1, showfliers=False)


    # palette = ['g'] * 5 + ['b'] * 5 + ['m'] * 1 + ['c'] * 2
    palette = ['m'] * 1 + ['g'] * 5 + ['b'] * 5 + ['c'] * 2
    markers = ['D'] * 1 + ['o'] * 5 + ['^'] * 5 + ['s'] * 2

    for x, val, c, m in zip(xs, vals, palette, markers):
        plt.scatter(x, val, alpha=0.5, color = c, zorder = 2, marker = m)
        
    if type_plot == 'FS_FIP':
        plt.plot((0,16),(new_original_FIPs,new_original_FIPs), linestyle = '--', c = 'k', linewidth=1.0, zorder = 1, label = 'Maximum FS FIP in unaltered $72^{3}$ microstructure')
    else:
        plt.plot((0,16),(new_original_FIPs,new_original_FIPs), linestyle = '--', c = 'k', linewidth=1.0, zorder = 1, label = 'Maximum PSSR in unaltered $72^{3}$ microstructure')

    # Star markers for largest FIP in entire 1st NN results
    p1 = plt.scatter(xs_1[0], vals_1[0], alpha=0.65, marker = '^', color = 'r', s = 25, label = 'Largest in entire microstructure, not in GOI')

    # Star markers for largest FIP in entire 1st NN results
    p2 = plt.scatter(xs_1[1], vals_1[1], alpha=0.65, marker = '^', color = 'r', s = 25)

    # l = plt.legend([(p1, p2)], ['Largest FIPs in microstructure'], numpoints=1, fontsize = 6, loc = 'upper right', handler_map={tuple: HandlerTuple(ndivide=None)})

    plt.legend(framealpha = 1, loc = 'lower left', fontsize = fontsizee)
    plt.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0), useMathText = True)



    if type_plot == 'FS_FIP':
        plt.ylim(0.0036, 0.014)
    elif type_plot == 'PSSR':
        plt.ylim(0.00058, 0.00222)
    plt.xlim(0.5,13.5)
    
    if type_plot == 'FS_FIP':
        plt.ylabel('Largest sub-band averaged FS FIP in grain 2424')
    elif type_plot == 'PSSR':
        plt.ylabel('Largest sub-band averaged PSSR in grain 2424')
    plt.xlabel('Altered grain orientations')    
    plt.grid(True, zorder = 1, axis = 'y', alpha = 0.4)
    plt.xticks(rotation = 30, fontsize = fontsizee)
    plt.tight_layout()
    
    if type_plot == 'FS_FIP':
        plt.savefig(os.path.join(store_dirr, '1st_highest_FS_FIP_variation_first_plot'))
    elif type_plot == 'PSSR':
        plt.savefig(os.path.join(store_dirr, '1st_highest_PSSR_variation_first_plot'))
    plt.close()

def plot_COVs():
    # Function to plot coefficient of variation for the FS_FIP, PSSR, and maximum normal stresses

    
    ''' Read in previously written data for FS FIP '''
    
    fname = '1st_highest_FS_FIP_variation_first_plot_data.p'
    h1 = open(os.path.join(store_dirr, fname), 'rb')
    FS_FIP_dataset1, FS_FIP_dataset2, FS_FIP_names1 = p.load(h1)
    h1.close()
    
    FS_FIP_COVs = np.zeros(11)
    
    for qqq in range(11):
        FS_FIP_COVs[qqq] = scipy.stats.variation(FS_FIP_dataset1[qqq]) * 100



    ''' Read in previously written data for PSSR '''
    
    fname = '1st_highest_PSSR_variation_first_plot_data.p'
    h1 = open(os.path.join(store_dirr, fname), 'rb')
    PSSR_dataset1, PSSR_dataset2, PSSR_names1 = p.load(h1)
    h1.close()

    
    PSSR_COVs = np.zeros(11)
    
    for qqq in range(11):
        PSSR_COVs[qqq] = scipy.stats.variation(PSSR_dataset1[qqq]) * 100



    ''' Read in previously written data or compile data for normal stress '''
    
    fname = '1st_highest_normal_stress_variation_first_plot_data.p'
    
    if os.path.exists(os.path.join(store_dirr, fname)):
        h1 = open(os.path.join(store_dirr, fname), 'rb')
        normal_stress_dataset1 = p.load(h1)
        h1.close()
    else:
        normal_stress_dataset1 = read_normal_stress_for_COV()

    
    normal_stress_COVs = np.zeros(11)
    
    for qqq in range(11):
        normal_stress_COVs[qqq] = scipy.stats.variation(normal_stress_dataset1[qqq]) * 100



    # Plot COVs

    # covs = np.zeros(11)

    # for qqq in range(11):
    #     covs[qqq] = scipy.stats.variation(dataset1[qqq])
        

    # sigma_COV = [0.56, 0.34, 0.16, 0.21, 0.02, 0.00, 1.89, 0.88, 0.61, 0.13, 0.21]
    # FIP_COV   = [7.96, 7.60, 0.86, 1.78, 0.08, 0.21, 4.62, 2.40, 7.55, 7.17, 2.49]
    # PSSR_COV  = [7.50, 6.45, 0.89, 1.85, 0.08, 0.21, 8.05, 1.33, 7.31, 7.23, 2.39]

    all_COV = [PSSR_COVs, normal_stress_COVs, FS_FIP_COVs]
    all_COV_t = np.transpose(all_COV)

    names1 = ['$1^{st}$ NN with most SA', 
              'Largest $1^{st}$ NN','Largest $2^{nd}$ NN', 'Largest $3^{rd}$ NN', 'Largest $4^{th}$ NN', 'Largest $5^{th}$ NN', 
              'All 13 $1^{st}$ NNs','All 68 $2^{nd}$ NNs', 'All 159 $3^{rd}$ NNs', 'All 303 $4^{th}$ NNs', 'All 517 $5^{th}$ NNs']

    fontsizee = 7


    fig = plt.figure(facecolor="white", figsize=(5, 4), dpi=1200)
    plt.rcParams["mathtext.default"] = "regular"

    plt_labels = ['PSSR', r"$\sigma_n$", 'FIP']
    markers = ['s', 'o', '^']
    colors = ['g', 'm', 'b']

    for ii in range(3):
        plt.scatter(range(1, 12), all_COV[ii], label = plt_labels[ii], zorder = 2, c = colors[ii], marker = markers[ii])

    # plt.plot((1.5,1.5), (-1,10), linestyle = '--', c = 'k', linewidth = 0.45)
    # plt.plot((6.5,6.5), (-1,10), linestyle = '--', c = 'k', linewidth = 0.45)

    plt.legend(framealpha = 1)
    plt.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0), useMathText = True)

    # plt.ylim(-0.35, 8.4)
    plt.ylabel('Coefficient of Variation [%]')    
    plt.xlabel('Altered grain orientations')    
    plt.grid(True, zorder = 1, axis = 'y', alpha = 0.4)
    plt.xticks(range(1, 12), labels = names1, rotation = 30, fontsize = 6)
    plt.tight_layout()
    plt.savefig(os.path.join(store_dirr, '1st_highest_GOI_coef_of_variation'))
    plt.close()

def read_and_plot_secondary_FIPs(file_name_template):
    # This function will read in the FIPs to create the second plot for the grain that manifests the 1st highest FIP in the 160,000 grain microstructure

    # 5
    ''' Change 5 in 3rd layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\Layer3_Largest_5_Grain')
    num_inst = 5

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_5_in_layer_3 = []


    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        grain_2424_change_5_in_layer_3.append(total_all_data[ii].transpose()[0][kk])

    new_largest_change_5_in_layer_3 = []
    for tt in total_all_data:
        new_largest_change_5_in_layer_3.append(tt[0][0])

    grain_with_new_largest_FIP_change_5_in_layer_3 = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_change_5_in_layer_3.append(tt[0])

    # 5
    ''' Change 5% in 3rd layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\Layer3_Largest_5Percent_Grain')
    num_inst = 5

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_5_perc_in_layer_3 = []


    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        grain_2424_change_5_perc_in_layer_3.append(total_all_data[ii].transpose()[0][kk])

    new_largest_change_5_perc_in_layer_3 = []
    for tt in total_all_data:
        new_largest_change_5_perc_in_layer_3.append(tt[0][0])

    grain_with_new_largest_FIP_change_5_perc_in_layer_3 = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_change_5_perc_in_layer_3.append(tt[0])

    # 5
    ''' Change 20% in 3rd layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\Layer3_Largest_20Percent_Grain')
    num_inst = 5

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_20_perc_in_layer_3 = []


    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        grain_2424_change_20_perc_in_layer_3.append(total_all_data[ii].transpose()[0][kk])

    new_largest_change_20_perc_in_layer_3 = []
    for tt in total_all_data:
        new_largest_change_20_perc_in_layer_3.append(tt[0][0])

    grain_with_new_largest_FIP_change_20_perc_in_layer_3 = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_change_20_perc_in_layer_3.append(tt[0])

    # 5
    ''' Change 5 in 4th layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\Layer4_Largest_5_Grain')
    num_inst = 5

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_5_in_layer_4 = []


    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        grain_2424_change_5_in_layer_4.append(total_all_data[ii].transpose()[0][kk])

    new_largest_change_5_in_layer_4 = []
    for tt in total_all_data:
        new_largest_change_5_in_layer_4.append(tt[0][0])

    grain_with_new_largest_FIP_change_5_in_layer_4 = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_change_5_in_layer_4.append(tt[0])

    # 5
    ''' Change 5% in 4th layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\Layer4_Largest_5Percent_Grain')
    num_inst = 5

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_5_perc_in_layer_4 = []


    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        grain_2424_change_5_perc_in_layer_4.append(total_all_data[ii].transpose()[0][kk])

    new_largest_change_5_perc_in_layer_4 = []
    for tt in total_all_data:
        new_largest_change_5_perc_in_layer_4.append(tt[0][0])

    grain_with_new_largest_FIP_change_5_perc_in_layer_4 = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_change_5_perc_in_layer_4.append(tt[0])

    # 5
    ''' Change 20% in 4th layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\Layer4_Largest_20Percent_Grain')
    num_inst = 5

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_20_perc_in_layer_4 = []


    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        grain_2424_change_20_perc_in_layer_4.append(total_all_data[ii].transpose()[0][kk])

    new_largest_change_20_perc_in_layer_4 = []
    for tt in total_all_data:
        new_largest_change_20_perc_in_layer_4.append(tt[0][0])

    grain_with_new_largest_FIP_change_20_perc_in_layer_4 = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_change_20_perc_in_layer_4.append(tt[0])
    
    # 5
    ''' Change 5 in 5th layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\Layer5_Largest_5_Grain')
    num_inst = 5

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_5_in_layer_5 = []


    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        grain_2424_change_5_in_layer_5.append(total_all_data[ii].transpose()[0][kk])

    new_largest_change_5_in_layer_5 = []
    for tt in total_all_data:
        new_largest_change_5_in_layer_5.append(tt[0][0])

    grain_with_new_largest_FIP_change_5_in_layer_5 = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_change_5_in_layer_5.append(tt[0])

    # 5
    ''' Change 5% in 5th layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\Layer5_Largest_5Percent_Grain')
    num_inst = 5

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_5_perc_in_layer_5 = []


    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        grain_2424_change_5_perc_in_layer_5.append(total_all_data[ii].transpose()[0][kk])

    new_largest_change_5_perc_in_layer_5 = []
    for tt in total_all_data:
        new_largest_change_5_perc_in_layer_5.append(tt[0][0])

    grain_with_new_largest_FIP_change_5_perc_in_layer_5 = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_change_5_perc_in_layer_5.append(tt[0])

    # 5
    ''' Change 20% in 5th layer '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\Layer5_Largest_20Percent_Grain')
    num_inst = 5

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_change_20_perc_in_layer_5 = []


    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        grain_2424_change_20_perc_in_layer_5.append(total_all_data[ii].transpose()[0][kk])

    new_largest_change_20_perc_in_layer_5 = []
    for tt in total_all_data:
        new_largest_change_20_perc_in_layer_5.append(tt[0][0])

    grain_with_new_largest_FIP_change_20_perc_in_layer_5 = []
    for tt in total_added_g:
        grain_with_new_largest_FIP_change_20_perc_in_layer_5.append(tt[0])


    # 1
    ''' COMPARE TO ORIGINAL FROM 250^3 '''
    directory = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_250^3\data\cropped_to_72_72_72_original_orientations')
    num_inst = 1

    total_fips = []
    total_all_data = []
    total_added_g  = []

    # NOTE: 'all_data' is indexed at 0!

    for ii in range(num_inst):
        print('On instantiation %d.' % ii)
        new_all_fs_fips, all_data, added_g = read_FIPs_from_certain_grain(directory, file_name_template, ii)

        total_fips.append(new_all_fs_fips)
        total_all_data.append(all_data)
        total_added_g.append(added_g)


    find_2424 = []
    for vals in total_all_data:
    # Find grain 2424 ...
        find_2424.append(np.where(vals.transpose()[1] == grain_num_of_interest)[0][0])


    grain_2424_original = []
    for ii, kk in enumerate(find_2424):
        # print(ii,kk)
        
        grain_2424_original.append(total_all_data[ii].transpose()[0][kk])

    new_original_FIPs = []
    for tt in total_all_data:
        new_original_FIPs.append(tt[0][0])




    # https://towardsdatascience.com/scattered-boxplots-graphing-experimental-results-with-matplotlib-seaborn-and-pandas-81f9fa8a1801


    dataset1 = [grain_2424_change_5_in_layer_3, grain_2424_change_5_in_layer_4, grain_2424_change_5_in_layer_5,
                grain_2424_change_5_perc_in_layer_3, grain_2424_change_5_perc_in_layer_4, grain_2424_change_5_perc_in_layer_5,
                grain_2424_change_20_perc_in_layer_3, grain_2424_change_20_perc_in_layer_4, grain_2424_change_20_perc_in_layer_5]

    names1 = ['Largest 5 in $3^{rd}$ layer',   'Largest 5 in $4^{th}$ layer',   'Largest 5 in $5^{th}$ layer',
              'Largest 5% in $3^{rd}$ layer',  'Largest 5% in $4^{th}$ layer',  'Largest 5% in $5^{th}$ layer',
              'Largest 20% in $3^{rd}$ layer', 'Largest 20% in $4^{th}$ layer', 'Largest 20% in $5^{th}$ layer']


    vals, names, xs = [],[],[]

    for i, col in enumerate(names1):
        vals.append(dataset1[i])
        names.append(col)
        xs.append(np.random.normal(i + 1, 0.04, len(dataset1[i])))  # adds jitter to the data points - can be adjusted


    # dataset2 = [new_largest_ALL_first_NN_new_FIPs, new_largest_ALL_second_NN_new_FIPs]
    # vals_1, xs_1 = [], []

    # for kk in range(2):
    #     vals_1.append(dataset2[kk])
    #     xs_1.append(np.random.normal(kk + 7, 0.04, len(dataset2[kk])))  # adds jitter to the data points - can be adjusted
        
        
    # Store FIPs pickle file!

    fname = '1st_highest_FS_FIP_variation_second_plot_data.p'
    h1 = open(os.path.join(store_dirr, fname), 'wb')
    p.dump([dataset1, names1], h1)
    h1.close()


    fontsizee = 7

    fig = plt.figure(facecolor="white", figsize=(7.5, 4.5), dpi=1200)
    plt.rcParams["mathtext.default"] = "regular"
    plt.boxplot(vals[0:9], labels=names, zorder = 1, showfliers=False)


    # palette = ['g'] * 5 + ['b'] * 5 + ['m'] * 1 + ['c'] * 2
    # palette = ['m'] * 1 + ['g'] * 5 + ['b'] * 5 + ['c'] * 2
    markers = ['p'] * 3 + ['d'] * 3 + ['*'] * 3
    palette = ['darkmagenta', 'crimson', 'deepskyblue', 'darkmagenta', 'crimson', 'deepskyblue', 'darkmagenta', 'crimson', 'deepskyblue']
    palette = ['m', 'r', 'b', 'm', 'r', 'b', 'm', 'r', 'b']

    for x, val, c, m in zip(xs, vals, palette, markers):
        plt.scatter(x, val, alpha=0.55, color = c, zorder = 2, marker = m)
        
    plt.plot((0,16),(new_original_FIPs,new_original_FIPs), linestyle = '--', c = 'k', linewidth=1.0, zorder = 1, label = 'Maximum FIP in unaltered $72^{3}$ microstructure')

    # Star markers for largest FIP in entire 1st NN results
    # p1 = plt.scatter(xs_1[0], vals_1[0], alpha=0.65, marker = '^', color = 'r', s = 25, label = 'Largest in entire microstructure, not in GOI')

    # Star markers for largest FIP in entire 1st NN results
    # p2 = plt.scatter(xs_1[1], vals_1[1], alpha=0.65, marker = '^', color = 'r', s = 25)

    # l = plt.legend([(p1, p2)], ['Largest FIPs in microstructure'], numpoints=1, fontsize = 6, loc = 'upper right', handler_map={tuple: HandlerTuple(ndivide=None)})

    plt.legend(framealpha = 1, loc = 'lower left', fontsize = fontsizee)

    plt.yticks(np.arange(1.1e-2, 1.3e-2, 0.3e-3))

    plt.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0), useMathText = True)

    # plt.ylim(0.0036, 0.014)
    plt.ylim(0.011, 0.013)
    plt.xlim(0.5,9.5)

    plt.ylabel('Largest sub-band averaged FS FIP in grain 2424')    
    plt.xlabel('Altered grain orientations')    
    plt.grid(True, zorder = 1, axis = 'y', alpha = 0.4)
    plt.xticks(rotation = 30, fontsize = fontsizee)
    plt.tight_layout()
    plt.savefig(os.path.join(store_dirr, '1st_highest_FS_FIP_variation_second_plot'))
    plt.close()

def main():
    # Plot variation in FIPs; first plot
    read_and_plot_main_FIP_PSSR(file_name_template_master[0], 'FS_FIP')
    
    # Plot variation in Plastic shear strain range; first plot
    read_and_plot_main_FIP_PSSR(file_name_template_master[2], 'PSSR')

    # Plot variation in FIPs; second plot
    read_and_plot_secondary_FIPs(file_name_template_master[0])
    
    # Plot coefficient of variation
    plot_COVs()

if __name__ == "__main__":
    main()