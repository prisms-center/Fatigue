import numpy as np
import os
import matplotlib.pyplot as plt
import re
import pickle as p
import linecache
import operator
from itertools import combinations
import time
import shutil
import fileinput
import subprocess
import pandas as pd
import time
from mpl_toolkits.mplot3d import Axes3D
import math
import xlrd

# Get name of directory that contains the PRISMS-Fatigue scripts
DIR_LOC = os.path.dirname(os.path.abspath(__file__))

# Define FCC octahedral slip plane normal directions
FCC_OCT_NORMALS = np.asarray([[1,1,1],
[-1,1,1],
[1,1,-1],
[1,-1,1]]) / (3**0.5)

# Define FCC octahedral slip directions
FCC_OCT_SLIP_DIRS = np.asarray([[0,1,-1],
[-1,0,1],
[1,-1,0],
[0,-1,1],

[-1,0,-1],
[1,1,0],
[0,-1,-1],
[1,0,1],

[-1,1,0],
[0,1,1],
[1,0,-1],
[-1,-1,0]]) / (2**0.5)


# Combine slip planes and slip directions for FCC and verify normality
FCC_SLIP_SYSTEMS = np.zeros((12,2,3))
for tt in range(12):
    FCC_SLIP_SYSTEMS[tt][0] = FCC_OCT_NORMALS[int(tt/3)]
    FCC_SLIP_SYSTEMS[tt][1] = FCC_OCT_SLIP_DIRS[tt]
    # print(np.dot(FCC_SLIP_SYSTEMS[tt][0],FCC_SLIP_SYSTEMS[tt][1]))

def read_data_from_160000_microstructure(directory):
    # Additional unused function to read in the largest sub-band averaged FIPs (one per grain) from a single microstructure instantiation
    # This is particularly useful when considering a very large SVE with more than ~10,000 grains as this function can take a long time!
    
    print('Read FIP data from the 160,000 grain microstructure')

    # Specify name of pickle file with sub-band averaged FIPs
    fname = os.path.join(directory, 'sub_band_averaged_FS_FIP_pickle_0.p')
    
    # Read in FIPs
    h1 = open(fname,'rb')
    fips = p.load(h1)
    h1.close()
    
    # Sort in descending order
    sorted_fips = sorted(fips.items(), key=operator.itemgetter(1))
    sorted_fips.reverse()    
    
    # Initialize list of just FIPs
    new_all_fs_fips = []

    # Initialize list to keep track of which grains have already been considered
    added_g = []
    
    # Initialize array with more detailed FIP data
    # FIP, grain number, slip system number, layer number, sub-band region number
    all_data = np.zeros(((len(sorted_fips)),5))
    
    for kk in range(len(sorted_fips)):
        new_all_fs_fips.append(sorted_fips[kk][1])
        
        all_data[kk][0] = sorted_fips[kk][1]
        all_data[kk][1] = sorted_fips[kk][0][0]
        all_data[kk][2] = sorted_fips[kk][0][1]
        all_data[kk][3] = sorted_fips[kk][0][2]
        all_data[kk][4] = sorted_fips[kk][0][3]  
    
        added_g.append(sorted_fips[kk][0][0])
    
    # Store all data
    fname = 'all_FIP_data_compiled.p'    
    h1 = open(os.path.join(directory, fname),'wb')
    p.dump([new_all_fs_fips, all_data, added_g],h1)
    h1.close()
    
    return new_all_fs_fips, all_data, added_g

def read_elements_of_highest_FIPs(directory, all_data, num_extract):

    # WARNING: This function is extremely memory intensive because of the very large size of the file 'sub_band_info_0.p' !!! 
    # This function reads in which elements belong to the sub bands that manifest the largest FIPs
    
    fname3 = os.path.join(directory, 'top_elem_fips.p')
    
    if not os.path.exists(fname3):

        # Read file that contains sub band element data
        fname = os.path.join(directory, 'sub_band_info_0.p')
        h1 = open(fname,'rb')
        master_sub_band_dictionary,number_of_layers,number_of_sub_bands = p.load(h1, encoding = 'latin1')
        h1.close()
        
        elems = []
        
        for kk in range(num_extract):
        
            # The values read in are indexed at 0!
            grain_temp    = all_data[kk][1]
            SS_temp       = int(all_data[kk][2]/3)
            layer_temp    = all_data[kk][3]
            sub_band_temp = all_data[kk][4]

            elems.append(master_sub_band_dictionary[grain_temp,SS_temp,layer_temp,sub_band_temp])
            
        h3 = open(fname3, 'wb')
        p.dump(elems,h3)
        h3.close()
        
    else:
        h3 = open(fname3, 'rb')
        elems = p.load(h3)
        h3.close()
    
    return elems

def read_FIP_components(directory):

    # Read components of the highest FIPs
    
    fname4 = os.path.join(directory, 'PRISMS_FIP_components_FS_FIP_0.p')
    h4 = open(fname4,'rb')
    plastic_shear_strain_range_FIP, normal_stresses_FIP = p.load(h4)
    h4.close()
    
    return plastic_shear_strain_range_FIP, normal_stresses_FIP

def flatten(d):
    return {i for b in [[i] if not isinstance(i, list) else flatten(i) for i in d] for i in b}
    
def EulerToMat(theta):
    # Calculates Rotation Matrix given euler angles.
    R = np.zeros((3,3))
    
    s1 = math.sin(theta[0])
    c1 = math.cos(theta[0])
    s2 = math.sin(theta[1])
    c2 = math.cos(theta[1])
    s3 = math.sin(theta[2])
    c3 = math.cos(theta[2])
    
    R[0,0] = c1*c3-s1*s3*c2
    R[1,0] = s1*c3+c1*s3*c2
    R[2,0] = s3*s2
    R[0,1] = -c1*s3-s1*c3*c2
    R[1,1] = -s1*s3+c1*c3*c2
    R[2,1] = c3*s2
    R[0,2] = s1*s2
    R[1,2] = -c1*s2
    R[2,2] = c2    
 
    return R    

def rotate_tensor(theta,tensor):
    # Rotates a tensor by Euler angles
    
    R   = EulerToMat(theta)
    R_t = np.transpose(R)
    
    tensor_rotated = np.matmul(R_t,np.matmul(tensor,R))
    return tensor_rotated
    
def rotate_vector(theta,vector):
    # Rotates a vector by Euler angles
    
    R   = EulerToMat(theta)
    R_t = np.transpose(R)
    
    tensor_rotated = np.matmul(vector,R_t)
    # tensor_rotated = np.matmul(R_t,vector)
    return tensor_rotated   

def calculate_FCC_Schmid_Factors(orientations, grains, load_dir):

    # Initialize array for calculated Schmid Factors
    SF_temp = np.ones((len(orientations),12))
    
    # Iterate through grains of interest
    for gg in grains:
    
        SF_temp[gg,:] = calculate_FCC_SF(orientations[gg],load_dir)

    return SF_temp[grains]
    
def calculate_FCC_SF(orientations,load_dir):

    # Rotate directions
    rotated_vec = rotate_vector(orientations,FCC_SLIP_SYSTEMS) 
    
    # Initialize array for calculated Schmid Factors
    temp_calc = np.zeros((12))
    
    # Iterate and calculate for each slip system
    for tt in range(12):
    
        temp_calc[tt] = np.abs( np.dot(rotated_vec[tt][0],load_dir) * np.dot(rotated_vec[tt][1],load_dir) ) 
    
    return temp_calc   

def top_50_FIPs_vtk():

    # Append a new scalar to the .vtk file to indicate grains that contain the highest 50 FIPs
    print('Appending top 50 FIP grains to .vtk file for visualization')
    directory = os.path.join(DIR_LOC, r'Section_4\160000_x_1')
    
    fips, all_data, added_g = read_data_from_160000_microstructure(directory)
    
    Fname_vtk_loc = os.path.join(directory, 'Output_FakeMatl_0_FIPs.vtk')
    Fname_vtk_new = os.path.join(directory, 'Output_FakeMatl_0_top_50_FIP_grains.vtk')
    
    # Create copy of original .vtk file in case something goes wrong!
    shutil.copy(Fname_vtk_loc, Fname_vtk_new)

    vtk_first_part = []
    with open(Fname_vtk_loc) as f:
        for line in f.readlines():
            vtk_first_part.append(line)
            if 'EulerAngles' in line:
                vtk_first_part.append('LOOKUP_TABLE default \n')
                break
    f.close()

    # Get just lines with grain IDs
    # NOTE: THESE NUMBERS BELOW WILL CHANGE BASED ON THE FIRST LINES IN THE .VTK FILE!!
    # FOR THE 160,000 GRAIN MICROSTRUCTURE, WE MUST OMIT THE FIRST 51 LINES, AS SHOWN BELOW
    grain_IDs = vtk_first_part[51:-2]

    ''' read in "highest_fips_and_grains.p" in the 160^3 results folder '''
    top_50_grains = all_data.transpose()[1].astype(int)[0:50]
    top_50 = [ii + 1 for ii in top_50_grains]

    f_vtk = open(Fname_vtk_new,'a')

    f_vtk.write('SCALARS Top_50_FIP_grains int 1\n')
    f_vtk.write('LOOKUP_TABLE default\n')

    counter = 0
    # Iterate through lines of grain IDs from .vtk file
    for kk in grain_IDs:

        # Get grain IDs
        temp_grain_IDs = [int(s) for s in kk.split() if s.isdigit()]
        
        # Iterate through each grain ID
        for jj in temp_grain_IDs:

            if jj in top_50:
                f_vtk.write(' 1')
            else:
                f_vtk.write(' 0')
            counter += 1 
            
            # Write new line
            if counter == 20:
                f_vtk.write('\n')
                counter = 0

    f_vtk.close()

def compute_correlations():

    # Define directory with the 160,000 grain microstructure discretized by 250^3 elements
    directory = os.path.join(DIR_LOC, r'Section_4\160000_x_1')
    
    # Define where to store plots
    store_dirr = os.path.join(DIR_LOC, r'plots')
    
    # Read in orientations based on DREAM.3D generated .csv file
    # The .csv file was edited to only contain the first set of data (i.e., data for each grain in row format)
    df1 = pd.read_csv(os.path.join(directory, 'FeatureData_FakeMatl_0_first_chunk.csv'), index_col = False)
    
    # Read in various properties of grains from microstructure
    orientations = df1[['EulerAngles_0','EulerAngles_1','EulerAngles_2']].to_numpy(dtype=float)
    diameters = df1[['EquivalentDiameters']].to_numpy(dtype=float)
    volumes = df1[['Volumes']].to_numpy(dtype=float)
    centroids = df1[['Centroids_0','Centroids_1','Centroids_2']].to_numpy(dtype=float)

    # Compile data from the 160,000 grain microstructure
    fips, all_data, added_g = read_data_from_160000_microstructure(directory)

    # Calculate Schmid Factors for grains in the 160,000 grain microstructure
    SFs = calculate_FCC_Schmid_Factors(orientations, added_g, [1,0,0])

    # Focus on the highset 250 FIPs
    num_fips_focus = 250
    SFs_250 = SFs[:num_fips_focus]

    # Determine the Schmid Factor of the slip system that manifests the highest FIP in each grain
    SF_of_highest_FIPs = np.zeros((num_fips_focus))
    
    for ii, vv in enumerate(all_data.transpose()[2][:num_fips_focus]):
        SF_of_highest_FIPs[ii] = SFs_250[ii][int(vv)]
    
    
    ''' Plot figures '''
    
    # Specify how many of the highest FIPs should be investigated
    top_fips_to_plot = 50
    
    # Plot Schmid Factors of the highest grains
    print('Plotting Schmid Factors')
    fig = plt.figure(facecolor="white", figsize=(6, 4), dpi=1200)
    cm = plt.cm.get_cmap('jet')
    plt.scatter(range(top_fips_to_plot), SF_of_highest_FIPs[0:top_fips_to_plot], c = all_data.transpose()[0][0:top_fips_to_plot], cmap=cm, zorder = 2)
    plt.grid(zorder = 1)
    cbarr = plt.colorbar()
    cbarr.set_label("Sub-band averaged FIP")

    plt.xlabel('Grain ID by FIP rank')
    plt.ylabel('Schmid Factor of Slip System with largest FIP')
    plt.tight_layout()
    plt.savefig(os.path.join(store_dirr, 'Schmid_Factor_by_grain_ID_FIP_rank_top_%d_FIPs' % top_fips_to_plot))
    plt.close()


    # Specify how many of the highest FIPs should be investigated
    top_fips_to_plot = 50

    # Plot grain diameters of the highest grains
    dia_highest_FIP_grains = np.zeros((num_fips_focus))
    
    for ii, vv in enumerate(all_data.transpose()[1][:num_fips_focus]):
        dia_highest_FIP_grains[ii] = diameters[int(vv)][0]
    
    print('Plotting grain diameters')
    # Plot the grain diameters of the grains that manifest the highest 50 FIPs
    fig = plt.figure(facecolor="white", figsize=(6, 4), dpi=1200)
    cm = plt.cm.get_cmap('jet')
    plt.scatter(range(top_fips_to_plot), dia_highest_FIP_grains[0:top_fips_to_plot], c = all_data.transpose()[0][0:top_fips_to_plot], cmap=cm, zorder = 2)
    plt.grid(zorder = 1)
    cbarr = plt.colorbar()
    cbarr.set_label("Sub-band averaged FIP")

    plt.xlabel('Grain ID by FIP rank')
    plt.ylabel('Equivalent grain diameter')
    plt.tight_layout()
    plt.savefig(os.path.join(store_dirr, 'Grain_diameter_by_grain_ID_FIP_rank_top_%d_FIPs' % top_fips_to_plot))
    plt.close()



    ''' Unused functions, left for prospective users '''
    # fig = plt.figure(facecolor="white", figsize=(7.5, 5), dpi=1200)
    # plt.scatter(SF_of_highest_FIPs[0:top_fips_to_plot], all_data.transpose()[0][0:top_fips_to_plot], zorder = 2)
    # plt.grid(zorder = 1)
    # plt.xlabel('Schmid Factor of Slip System with largest FIP')
    # plt.ylabel('FIP')
    # plt.tight_layout()
    # plt.savefig(os.path.join(store_dirr, 'FIPs_vs_Schmid_factor.png'))
    # plt.close()
    

    # fig = plt.figure(facecolor="white", figsize=(7.5, 5), dpi=1200)
    # cm = plt.cm.get_cmap('jet')
    # plt.scatter(SF_of_highest_FIPs[0:top_fips_to_plot], all_data.transpose()[0][0:top_fips_to_plot], zorder = 2, c = all_data.transpose()[0][0:top_fips_to_plot], cmap=cm)
    # plt.grid(zorder = 1)
    # plt.colorbar()
    # plt.xlabel('Schmid Factor of Slip System with largest FIP')
    # plt.ylabel('FIP')
    # plt.tight_layout()
    # plt.savefig(os.path.join(store_dirr, 'FIPs_vs_Schmid_factor_with_colored_FIPs.png'))
    # plt.close()
    
    
    
    # Read in the two components of the highset FIP to compute correlations
    plastic_shear_strain_range_FIP, normal_stresses_FIP = read_FIP_components(directory)

    # Read in the elements of the highest FIPs (not ALL FIPs to reduce memory demands)
    elems = read_elements_of_highest_FIPs(directory, all_data, num_fips_focus)
   

    # Calculate the plastic shear strain range and normal stresses over the same sub bands that manifest the highest FIPs in the 160,000 grain microstructure
    num_fips = 50
 
    FIP_plastic_range = np.zeros((num_fips))
    raw_normal_stress = np.zeros((num_fips))
    FIP_normal_stress = np.zeros((num_fips))
    FIP_recalculated  = np.zeros((num_fips))
    FIP_perc_diff     = np.zeros((num_fips))
    
    for ii in range(num_fips):
        
        FIP_plastic_range[ii] = np.mean(plastic_shear_strain_range_FIP[[ppp - 1 for ppp in elems[ii]]], axis = 0)[int(all_data[ii][2])]
        raw_normal_stress[ii] = np.mean( normal_stresses_FIP[[ppp - 1 for ppp in elems[ii]]], axis = 0)[int(all_data[ii][2])]
        FIP_normal_stress[ii] = np.mean((1 + 10 * (normal_stresses_FIP[[ppp - 1 for ppp in elems[ii]]] / 517)), axis = 0)[int(all_data[ii][2])]
        FIP_recalculated[ii]  = FIP_plastic_range[ii] * FIP_normal_stress[ii]
        FIP_perc_diff[ii]     = 100 * (FIP_recalculated[ii] - all_data[ii][0]) / all_data[ii][0]
    
    
    # Plot the plastic shear strain range of the sub bands that manifest the highest 50 FIPs
    print('Plotting plastic shear strain range')
    fig = plt.figure(facecolor="white", figsize=(6, 4), dpi=1200)
    cm = plt.cm.get_cmap('jet')
    plt.scatter(range(num_fips), FIP_plastic_range, c = all_data.transpose()[0][:num_fips], cmap=cm, zorder = 2)
    plt.grid(zorder = 1)
    plt.colorbar()

    plt.ylim(0.0011, 0.0021)
    plt.xlabel('Grain ID by FIP rank')
    plt.ylabel('PSSR on slip system with largest FIP')
    plt.tight_layout()
    plt.savefig(os.path.join(store_dirr, 'PSSR_by_grain_ID_FIP_rank_top_%d_FIPs' % num_fips))
    plt.close()   
    
    print('Plotting normal stresses')
    # Plot the normal stress of the sub bands that manifest the highest 50 FIPs
    fig = plt.figure(facecolor="white", figsize=(6, 4), dpi=1200)
    cm = plt.cm.get_cmap('jet')
    plt.scatter(range(num_fips), raw_normal_stress, c = all_data.transpose()[0][:num_fips], cmap=cm, zorder = 2)
    plt.grid(zorder = 1)
    plt.colorbar()
    
    plt.ylim(200, 310)
    plt.xlabel('Grain ID by FIP rank')
    plt.ylabel('Stress normal to slip system with largest FIP')
    plt.tight_layout()
    plt.savefig(os.path.join(store_dirr, 'Normal_stress_by_grain_ID_FIP_rank_top_%d_FIPs' % num_fips))
    plt.close()   
    
    
    
    
    
    
    
    
    
    
    
    

    ''' Read in list of neighbors '''

    # https://stackoverflow.com/questions/61885456/how-to-read-each-row-of-excel-file-into-a-list-so-as-to-make-whole-data-a-list-o
    listoflist = []
    workbook1 = xlrd.open_workbook(os.path.join(directory, r'FeatureData_FakeMatl_0_neighbor_list.xlsx'))
    b = workbook1.sheet_by_index(0)
    for i in range(0,b.nrows):
        rli = []
        for j in range(0,b.ncols):
            if b.cell_value(i,j) != '':
                rli.append(b.cell_value(i,j))
        listoflist.append(rli)
    
    neighbor_list = []   
    for kk in listoflist:
        neighbor_list.append([int(pp) for pp in kk[2:]])
        
        
    ''' Read in list of shared surface area! '''        
    
    listoflist = []
    workbook1 = xlrd.open_workbook(os.path.join(directory, r'FeatureData_FakeMatl_0_shared_surface_area_list.xlsx'))
    b = workbook1.sheet_by_index(0)
    for i in range(0,b.nrows):
        rli = []
        for j in range(0,b.ncols):
            if b.cell_value(i,j) != '':
                rli.append(b.cell_value(i,j))
        listoflist.append(rli)
    
    shared_surface_area_list = []   
    for kk in listoflist:
        shared_surface_area_list.append([pp for pp in kk[2:]])    
    
    
    # Let's examine the grains around the grain with the largest FIP
    largest_FIP_grain = int(all_data[0][1])
    nearest_neighbor_grains = [rr - 1 for rr in neighbor_list[largest_FIP_grain]]
    
    All_grain_SFs = calculate_FCC_Schmid_Factors(orientations, [x for x in range(len(orientations))], [1,0,0])
    
    
    ''' Unused functions, left for prospective users '''   
    # # CRUCIAL: GRAINS ARE ALSO ORIGINALLY INDEXED AT 1 !!!
    
    # SFs_of_grains_neighboring_largest_FIP_grain = calculate_FCC_Schmid_Factors(orientations, nearest_neighbor_grains, [1,0,0])
    
    # # Get largest Schmid Factor in the grains neighboring the largest FIP Grain
    
    # np.max(SFs_of_grains_neighboring_largest_FIP_grain, axis = 1)
    # np.mean(SFs_of_grains_neighboring_largest_FIP_grain, axis = 1)
    

    # num_neigh_highest_FIP = len(SFs_of_grains_neighboring_largest_FIP_grain)
    
    # # Same plot but different axes
    # fig, ax1 = plt.subplots()
    # ax1.plot(range(num_neigh_highest_FIP), shared_surface_area_list[84852], c = 'g', linestyle = ':', label = 'Shared surface area')
    # ax2 = ax1.twinx()
    # ax2.plot(range(num_neigh_highest_FIP), np.mean(SFs_of_grains_neighboring_largest_FIP_grain, axis = 1), c = 'r', label = 'Mean SF')
    # ax2.plot(range(num_neigh_highest_FIP), np.max(SFs_of_grains_neighboring_largest_FIP_grain, axis = 1), c = 'm', label = 'Max SF')     
    # fig.legend()
    # fig.tight_layout()
    # plt.show()
    # plt.close()
    
    
    
    # # "Lowest" possible SFs when pulling along the [111] direction
    # calculate_FCC_Schmid_Factors(np.zeros((2,3)),[0],[1,1,1]/np.sqrt(3))
    
    # # Calculate SF when straining along [111] plane
    
    # ori_1 = np.zeros((2,3))
    # ori_1[0] = np.asarray([0.97852153, 2.7551234 , 4.0642705 ])
    # # ori_1[0][0] = 45 * np.pi / 180
    
    # calculate_FCC_Schmid_Factors(ori_1,[0],[1,0,0])
    
    # # Plot averages as histogram
    
    # plt.hist(np.mean(All_grain_SFs, axis = 1), bins = 1000)
    # plt.show()
    # plt.close()
    
    
    # plt.hist(np.max(All_grain_SFs, axis = 1), bins = 1000)
    # plt.show()
    # plt.close()   










    ''' Misorientation analysis! '''
    
    # Read in misorientations calculated by DREAM.3D
    
    listoflist = []
    workbook1 = xlrd.open_workbook(os.path.join(directory, r'miso_list_rerun.xlsx'))
    b = workbook1.sheet_by_index(0)
    for i in range(0,b.nrows):
        rli = []
        for j in range(0,b.ncols):
            if b.cell_value(i,j) != '':
                rli.append(b.cell_value(i,j))
        listoflist.append(rli)
    
    miso_list = []   
    for kk in listoflist:
        miso_list.append([pp for pp in kk[2:]])    

    # Verify number of neighbors for each grain
    
    countt = 0
    for ii in range(len(miso_list)):
        if len(miso_list[0]) != len(shared_surface_area_list[0]):
            countt += 1 
            # IT IS VERIFIED


    ''' This section will plot all grain misorientations '''
    flat_miso_list = [item for sublist in miso_list for item in sublist]
    # plt.hist(flat_miso_list, bins = 1000)
    # plt.show()
    # plt.close()
    
    
    # The average of the minimum misorientation between each grain and its neighbors:
    avg_min_miso = np.zeros((len(miso_list)))
    for ii, miso_act in enumerate(miso_list):
        avg_min_miso[ii] = np.min(miso_act)
    # np.mean(avg_min_miso)

    
    miso_of_highest_grains = []
    avg_min_miso_of_highest_grains = np.zeros((num_fips))
    for ii in range(num_fips):
        miso_of_highest_grains.append(miso_list[int(all_data[ii][1])])
    
    for ii, miso_act in enumerate(miso_of_highest_grains):
        avg_min_miso_of_highest_grains[ii] = np.min(miso_act)
    # np.mean(avg_min_miso_of_highest_grains)



    # FIG A; plot misorientations
    print('Plotting correlations')
    fig = plt.figure(facecolor="white", figsize=(6, 4), dpi=1200)
    num_to_plot = 50
    plt.plot(range(1, num_to_plot+1), avg_min_miso_of_highest_grains[0:num_to_plot], c = 'b', zorder = 2, label = 'Highest %d FIP grains' % num_to_plot, marker = 'o', markersize = '4')
    plt.plot(range(1, num_to_plot+1), avg_min_miso[0:num_to_plot],  linestyle = ':', c = 'r', zorder = 2, label = 'First %d grains' % num_to_plot, marker = 's', markersize = '4')
    plt.grid(zorder = 1)
    plt.legend(framealpha = 1, fontsize = '9', loc = 'best')

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["mathtext.default"] = "regular"

    plt.xlabel('Grain ID by FIP rank')
    plt.ylabel('Lowest misorientation between $1^{st}$ NN [degrees]', fontsize = '9')
    plt.tight_layout()
    plt.savefig(os.path.join(store_dirr, 'Misorientations_by_grain_ID_FIP_rank_top_%d_FIPs' % num_fips))
    plt.close()  



    # FIG B; plot Schmid Factor correlation factor R1

    num_to_consider = 50
    
    average_of_max_neighborhood_SFs = np.zeros((num_to_consider))
    average_of_max_SF_of_all_grains = np.zeros((num_to_consider))
    
    # average_of_max_SF_of_all_grains = np.max(All_grain_SFs[0:num_to_consider], axis = 1)
    
    for ppp in range(num_to_consider):
    
        ''' Do for the highest FIPs first '''
        # Let's examine the grains around the grain with the largest FIP
        largest_FIP_grain = int(all_data[ppp][1])
        nearest_neighbor_grains = [rr - 1 for rr in neighbor_list[largest_FIP_grain]]

        # CRUCIAL: GRAINS ARE ALSO ORIGINALLY INDEXED AT 1 !!!
        
        SFs_of_grains_neighboring_largest_FIP_grain = calculate_FCC_Schmid_Factors(orientations, nearest_neighbor_grains, [1,0,0])
    
        # Get largest Schmid Factor in the grains neighboring the largest FIP Grain
        # np.max(SFs_of_grains_neighboring_largest_FIP_grain, axis = 1)
        # np.mean(SFs_of_grains_neighboring_largest_FIP_grain, axis = 1)
    
        num_neigh_highest_FIP = len(SFs_of_grains_neighboring_largest_FIP_grain)
        
        average_of_max_neighborhood_SFs[ppp] = np.mean(np.max(SFs_of_grains_neighboring_largest_FIP_grain, axis = 1))
        
        # average_of_max_neighborhood_SFs[ppp] = np.mean(SFs_of_grains_neighboring_largest_FIP_grain)
        
        
        
        ''' Now repeat for the first num_to_consider grains! '''
        
        nearest_neighbor_grains_first = [rr - 1 for rr in neighbor_list[ppp]]
        SFs_of_first_grains = calculate_FCC_Schmid_Factors(orientations, nearest_neighbor_grains_first, [1,0,0])
        # np.max(SFs_of_first_grains, axis = 1)
        num_neigh_highest_FIP = len(SFs_of_first_grains)
    
        average_of_max_SF_of_all_grains[ppp] = np.mean(np.max(SFs_of_first_grains, axis = 1))        
        
        # average_of_max_SF_of_all_grains[ppp] = np.mean(SFs_of_first_grains)
        
        
    fig = plt.figure(facecolor="white", figsize=(6, 4), dpi=1200)     
    plt.plot(range(1, num_to_plot+1), SF_of_highest_FIPs[0:num_to_consider]/average_of_max_neighborhood_SFs, c = 'b',  label = 'Highest %d FIP grains' % num_to_plot, zorder = 2, marker = 'o', markersize = '4') 
    plt.plot(range(1, num_to_plot+1), np.max(All_grain_SFs[0:num_to_consider],axis = 1)/average_of_max_SF_of_all_grains, c = 'r', label = 'First %d grains' % num_to_plot, zorder = 2, marker = 's', markersize = '4', linestyle = ':') 
    
    # plt.scatter(range(num_to_consider), SF_of_highest_FIPs[0:num_to_consider]/average_of_max_neighborhood_SFs, c = 'b',  label = '%d highest FIPs' % num_to_consider, zorder = 1, s = 2) 
    # plt.scatter(range(num_to_consider), np.max(All_grain_SFs[0:num_to_consider],axis = 1)/average_of_max_SF_of_all_grains, c = 'r',  label = '%d first grains' % num_to_consider, zorder = 1, s = 2)   
    
    plt.legend(framealpha = 1, fontsize = '9', loc = 'best')
    plt.grid('True', zorder = 1)
    # plt.ylim(0.9,1.25)
    
    plt.xlabel('Grain ID by FIP rank')
    plt.ylabel('Largest SF in grain / average of max $1^{st}$ NN SFs')
    
    # plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(store_dirr, 'Ratio_R1_by_grain_ID_FIP_rank_top_%d_FIPs' % num_to_consider))
    plt.close()           



    # FIG C; plot Schmid Factor correlation factor R2

    num_to_consider = 50
    
    average_of_max_neighborhood_SFs = np.zeros((num_to_consider))
    average_of_max_SF_of_all_grains = np.zeros((num_to_consider))
    
    # average_of_max_SF_of_all_grains = np.max(All_grain_SFs[0:num_to_consider], axis = 1)
    
    for ppp in range(num_to_consider):
    
        ''' Do for the highest FIPs first '''
        # Let's examine the grains around the grain with the largest FIP
        largest_FIP_grain = int(all_data[ppp][1])
        nearest_neighbor_grains = [rr - 1 for rr in neighbor_list[largest_FIP_grain]]

        # CRUCIAL: GRAINS ARE ALSO ORIGINALLY INDEXED AT 1 !!!
        
        SFs_of_grains_neighboring_largest_FIP_grain = calculate_FCC_Schmid_Factors(orientations, nearest_neighbor_grains, [1,0,0])
    
        # Get largest Schmid Factor in the grains neighboring the largest FIP Grain
        # np.max(SFs_of_grains_neighboring_largest_FIP_grain, axis = 1)
        # np.mean(SFs_of_grains_neighboring_largest_FIP_grain, axis = 1)
    
        num_neigh_highest_FIP = len(SFs_of_grains_neighboring_largest_FIP_grain)
        
        # average_of_max_neighborhood_SFs[ppp] = np.mean(np.max(SFs_of_grains_neighboring_largest_FIP_grain, axis = 1))
        
        average_of_max_neighborhood_SFs[ppp] = np.mean(SFs_of_grains_neighboring_largest_FIP_grain)
        
        
        
        ''' Now repeat for the first num_to_consider grains! '''
        
        nearest_neighbor_grains_first = [rr - 1 for rr in neighbor_list[ppp]]
        SFs_of_first_grains = calculate_FCC_Schmid_Factors(orientations, nearest_neighbor_grains_first, [1,0,0])
        # np.max(SFs_of_first_grains, axis = 1)
        num_neigh_highest_FIP = len(SFs_of_first_grains)
    
        # average_of_max_SF_of_all_grains[ppp] = np.mean(np.max(SFs_of_first_grains, axis = 1))        
        
        average_of_max_SF_of_all_grains[ppp] = np.mean(SFs_of_first_grains)
        
        
    fig = plt.figure(facecolor="white", figsize=(6, 4), dpi=1200)     
    plt.plot(range(1, num_to_plot+1), SF_of_highest_FIPs[0:num_to_consider]/average_of_max_neighborhood_SFs, c = 'b',  label = 'Highest %d FIP grains' % num_to_plot, zorder = 2, marker = 'o', markersize = '4') 
    plt.plot(range(1, num_to_plot+1), np.max(All_grain_SFs[0:num_to_consider],axis = 1)/average_of_max_SF_of_all_grains, c = 'r', label = 'First %d grains' % num_to_plot, zorder = 2, marker = 's', markersize = '4', linestyle = ':') 
    
    # plt.scatter(range(num_to_consider), SF_of_highest_FIPs[0:num_to_consider]/average_of_max_neighborhood_SFs, c = 'b',  label = '%d highest FIPs' % num_to_consider, zorder = 1, s = 2) 
    # plt.scatter(range(num_to_consider), np.max(All_grain_SFs[0:num_to_consider],axis = 1)/average_of_max_SF_of_all_grains, c = 'r',  label = '%d first grains' % num_to_consider, zorder = 1, s = 2)   
    
    plt.legend(framealpha = 1, fontsize = '9', loc = 'best')
    plt.grid('True', zorder = 1)
    plt.ylim(1.4, 2.7)
    
    plt.xlabel('Grain ID by FIP rank')
    plt.ylabel('Largest SF in grain / average of $1^{st}$ NN SFs')
    
    # plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(store_dirr, 'Ratio_R2_by_grain_ID_FIP_rank_top_%d_FIPs' % num_to_consider))
    plt.close()       



    # FIG D; plot Schmid Factor correlation factor R3

    num_to_consider = 50
    
    average_of_max_neighborhood_SFs = np.zeros((num_to_consider))
    average_of_max_SF_of_all_grains = np.zeros((num_to_consider))
    
    # average_of_max_SF_of_all_grains = np.max(All_grain_SFs[0:num_to_consider], axis = 1)
    
    for ppp in range(num_to_consider):
    
        ''' Do for the highest FIPs first '''
        # Let's examine the grains around the grain with the largest FIP
        largest_FIP_grain = int(all_data[ppp][1])
        nearest_neighbor_grains = [rr - 1 for rr in neighbor_list[largest_FIP_grain]]
        
        second_nearest_neighbor_grains = []
        
        for neighbors_1 in nearest_neighbor_grains:
            second_nearest_neighbor_grains.append(  [rr - 1 for rr in neighbor_list[neighbors_1]]   )

        flat_2nd_neighbors = list(flatten(second_nearest_neighbor_grains))
       

         
        # CRUCIAL: GRAINS ARE ALSO ORIGINALLY INDEXED AT 1 !!!
        
        SFs_of_grains_neighboring_largest_FIP_grain = calculate_FCC_Schmid_Factors(orientations, flat_2nd_neighbors, [1,0,0])
    
        # Get largest Schmid Factor in the grains neighboring the largest FIP Grain
        # np.max(SFs_of_grains_neighboring_largest_FIP_grain, axis = 1)
        # np.mean(SFs_of_grains_neighboring_largest_FIP_grain, axis = 1)
    
        num_neigh_highest_FIP = len(SFs_of_grains_neighboring_largest_FIP_grain)
        

        average_of_max_neighborhood_SFs[ppp] = np.mean(SFs_of_grains_neighboring_largest_FIP_grain)
        
        # print(num_neigh_highest_FIP)
        ''' Now repeat for the first 100 grains! '''
        
        nearest_neighbor_grains_first = [rr - 1 for rr in neighbor_list[ppp]]
        
        
        second_nearest_neighbor_grains_first_X_grains = []
        
        for neighbors_1 in nearest_neighbor_grains_first:
            second_nearest_neighbor_grains_first_X_grains.append(  [rr - 1 for rr in neighbor_list[neighbors_1]]   )

        flat_2nd_neighbors_first_x_Grains = list(flatten(second_nearest_neighbor_grains_first_X_grains))            

        
        SFs_of_first_grains = calculate_FCC_Schmid_Factors(orientations, flat_2nd_neighbors_first_x_Grains, [1,0,0])
        # np.max(SFs_of_first_grains, axis = 1)
        num_neigh_highest_FIP = len(SFs_of_first_grains)
    

        average_of_max_SF_of_all_grains[ppp] = np.mean(SFs_of_first_grains)
        
        
    fig = plt.figure(facecolor="white", figsize=(6, 4), dpi=1200)     
    plt.plot(range(1, num_to_plot+1), SF_of_highest_FIPs[0:num_to_consider]/average_of_max_neighborhood_SFs, c = 'b',  label = 'Highest %d FIP grains' % num_to_plot, zorder = 2, marker = 'o', markersize = '4') 
    plt.plot(range(1, num_to_plot+1), np.max(All_grain_SFs[0:num_to_consider],axis = 1)/average_of_max_SF_of_all_grains, c = 'r', label = 'First %d grains' % num_to_plot, zorder = 2, marker = 's', markersize = '4', linestyle = ':') 
    
    # plt.scatter(range(num_to_consider), SF_of_highest_FIPs[0:num_to_consider]/average_of_max_neighborhood_SFs, c = 'b',  label = '%d highest FIPs' % num_to_consider, zorder = 1, s = 2) 
    # plt.scatter(range(num_to_consider), np.max(All_grain_SFs[0:num_to_consider],axis = 1)/average_of_max_SF_of_all_grains, c = 'r',  label = '%d first grains' % num_to_consider, zorder = 1, s = 2)   
    
    plt.legend(framealpha = 1, fontsize = '9', loc = 'best')
    plt.grid('True', zorder = 2)
    plt.ylim(1.4, 2.7)
    
    plt.xlabel('Grain ID by FIP rank')
    plt.ylabel('Largest SF in grain / average of $1^{st}$ and $2^{nd}$ NN SFs', fontsize = '9')

    plt.tight_layout()
    plt.savefig(os.path.join(store_dirr, 'Ratio_R3_by_grain_ID_FIP_rank_top_%d_FIPs' % num_to_consider))
    plt.close()  










    plot_extra_info = False
    
    # This section has additional functions/code available to prospective users
    # These were written to further investigate spatial correlations between grains and the highest computed FIPs
    if plot_extra_info:

        ''' IMPORTANT!!! : In this section, consider the average values of all SFs in the neighbors! Normalize by shared surface area later! ''' 
        
        num_to_consider = 50
        
        average_of_max_neighborhood_SFs = np.zeros((num_to_consider))
        average_of_max_SF_of_all_grains = np.zeros((num_to_consider))
        
        # average_of_max_SF_of_all_grains = np.max(All_grain_SFs[0:num_to_consider], axis = 1)
        
        for ppp in range(num_to_consider):
        
            ''' Do for the highest FIPs first '''
            # Let's examine the grains around the grain with the largest FIP
            largest_FIP_grain = int(all_data[ppp][1])
            nearest_neighbor_grains = [rr - 1 for rr in neighbor_list[largest_FIP_grain]]

            # CRUCIAL: GRAINS ARE ALSO ORIGINALLY INDEXED AT 1 !!!
            
            SFs_of_grains_neighboring_largest_FIP_grain = calculate_FCC_Schmid_Factors(orientations, nearest_neighbor_grains, [1,0,0])
        
            # Get largest Schmid Factor in the grains neighboring the largest FIP Grain
            # np.max(SFs_of_grains_neighboring_largest_FIP_grain, axis = 1)
            # np.mean(SFs_of_grains_neighboring_largest_FIP_grain, axis = 1)
        
            num_neigh_highest_FIP = len(SFs_of_grains_neighboring_largest_FIP_grain)
            
            # average_of_max_neighborhood_SFs[ppp] = np.mean(np.max(SFs_of_grains_neighboring_largest_FIP_grain, axis = 1))
            
            average_of_max_neighborhood_SFs[ppp] = np.mean(SFs_of_grains_neighboring_largest_FIP_grain)
            
            
            
            ''' Now repeat for the first 100 grains! '''
            
            nearest_neighbor_grains_first = [rr - 1 for rr in neighbor_list[ppp]]
            SFs_of_first_grains = calculate_FCC_Schmid_Factors(orientations, nearest_neighbor_grains_first, [1,0,0])
            # np.max(SFs_of_first_grains, axis = 1)
            num_neigh_highest_FIP = len(SFs_of_first_grains)
        
            # average_of_max_SF_of_all_grains[ppp] = np.mean(np.max(SFs_of_first_grains, axis = 1))        
            
            average_of_max_SF_of_all_grains[ppp] = np.mean(SFs_of_first_grains)
            
            
        fig = plt.figure(facecolor="white", figsize=(7.5, 5), dpi=1200)     
        plt.plot(range(num_to_consider), SF_of_highest_FIPs[0:num_to_consider]/average_of_max_neighborhood_SFs, c = 'b', linestyle = ':',  label = '%d highest FIPs' % num_to_consider, zorder = 1) 
        plt.plot(range(num_to_consider), np.max(All_grain_SFs[0:num_to_consider],axis = 1)/average_of_max_SF_of_all_grains, c = 'r', linestyle = '-.', label = '%d first grains' % num_to_consider, zorder = 1) 
        
        # plt.scatter(range(num_to_consider), SF_of_highest_FIPs[0:num_to_consider]/average_of_max_neighborhood_SFs, c = 'b',  label = '%d highest FIPs' % num_to_consider, zorder = 1, s = 2) 
        # plt.scatter(range(num_to_consider), np.max(All_grain_SFs[0:num_to_consider],axis = 1)/average_of_max_SF_of_all_grains, c = 'r',  label = '%d first grains' % num_to_consider, zorder = 1, s = 2)   
        
        plt.legend(framealpha = 1)
        plt.grid('True', zorder = 2)
        # plt.ylim(0.9,1.25)
        
        plt.xlabel('Grain ID by FIP rank')
        plt.ylabel('Largest SF in grain / average of neighbors SFs')
        
        # plt.show()
        plt.tight_layout()
        plt.savefig(os.path.join(store_dirr, 'SF_of_neighboring_grains___average_SFs_of_neighbors_%d_FIPs' % num_to_consider))
        plt.close()           
        
        
        
        ''' Normalize by shared surface area '''
        
        num_to_consider = 50
        
        average_of_max_neighborhood_SFs = np.zeros((num_to_consider))
        average_of_max_SF_of_all_grains = np.zeros((num_to_consider))
        
        # average_of_max_SF_of_all_grains = np.max(All_grain_SFs[0:num_to_consider], axis = 1)
        
        for ppp in range(num_to_consider):
        
            ''' Do for the highest FIPs first '''
            # Let's examine the grains around the grain with the largest FIP
            largest_FIP_grain = int(all_data[ppp][1])
            nearest_neighbor_grains = [rr - 1 for rr in neighbor_list[largest_FIP_grain]]

            # CRUCIAL: GRAINS ARE ALSO ORIGINALLY INDEXED AT 1 !!!
            
            SFs_of_grains_neighboring_largest_FIP_grain = calculate_FCC_Schmid_Factors(orientations, nearest_neighbor_grains, [1,0,0])
        
            # Get largest Schmid Factor in the grains neighboring the largest FIP Grain
            # np.max(SFs_of_grains_neighboring_largest_FIP_grain, axis = 1)
            # np.mean(SFs_of_grains_neighboring_largest_FIP_grain, axis = 1)
        
            num_neigh_highest_FIP = len(SFs_of_grains_neighboring_largest_FIP_grain)
            
            # average_of_max_neighborhood_SFs[ppp] = np.mean(np.max(SFs_of_grains_neighboring_largest_FIP_grain, axis = 1))
            
            ''' Get shared area for this grain's neighbors: '''
            
            shared_area_btw_grain_and_neighbors = shared_surface_area_list[largest_FIP_grain]
            total_area = np.sum(shared_area_btw_grain_and_neighbors)
            
            normalized_SFs = np.zeros((num_neigh_highest_FIP,12))
            
            for ii in range(num_neigh_highest_FIP):
                normalized_SFs[ii] = (SFs_of_grains_neighboring_largest_FIP_grain[ii] * shared_area_btw_grain_and_neighbors[ii]) / total_area
            
            average_of_max_neighborhood_SFs[ppp] = np.mean(normalized_SFs)
            
            
            
            ''' Now repeat for the first 100 grains! '''
            
            nearest_neighbor_grains_first = [rr - 1 for rr in neighbor_list[ppp]]
            SFs_of_first_grains = calculate_FCC_Schmid_Factors(orientations, nearest_neighbor_grains_first, [1,0,0])
            num_neigh_highest_FIP_first_x_grains = len(SFs_of_first_grains)
            
            
            shared_area_btw_first_x_Grains = shared_surface_area_list[ppp]
            total_area_first_x_Grains = np.sum(shared_area_btw_first_x_Grains)
            
            normalized_SFs_first_x_grains = np.zeros((num_neigh_highest_FIP_first_x_grains,12))
        
            for ii in range(num_neigh_highest_FIP_first_x_grains):
                normalized_SFs_first_x_grains[ii] = (SFs_of_first_grains[ii] * shared_area_btw_first_x_Grains[ii]) / total_area_first_x_Grains
            
            average_of_max_SF_of_all_grains[ppp] = np.mean(normalized_SFs_first_x_grains)
            
            
        fig = plt.figure(facecolor="white", figsize=(7.5, 5), dpi=1200)     
        plt.plot(range(num_to_consider), SF_of_highest_FIPs[0:num_to_consider]/average_of_max_neighborhood_SFs, c = 'b', linestyle = ':',  label = '%d highest FIPs' % num_to_consider, zorder = 1) 
        plt.plot(range(num_to_consider), np.max(All_grain_SFs[0:num_to_consider],axis = 1)/average_of_max_SF_of_all_grains, c = 'r', linestyle = '-.', label = '%d first grains' % num_to_consider, zorder = 1) 
        
        # plt.scatter(range(num_to_consider), SF_of_highest_FIPs[0:num_to_consider]/average_of_max_neighborhood_SFs, c = 'b',  label = '%d highest FIPs' % num_to_consider, zorder = 1, s = 2) 
        # plt.scatter(range(num_to_consider), np.max(All_grain_SFs[0:num_to_consider],axis = 1)/average_of_max_SF_of_all_grains, c = 'r',  label = '%d first grains' % num_to_consider, zorder = 1, s = 2)   
        
        plt.legend(framealpha = 1)
        plt.grid('True', zorder = 2)
        # plt.ylim(0.9,1.25)
        
        plt.xlabel('Grain ID by FIP rank')
        plt.ylabel('Largest SF in grain / surface area weighted average of neighbors SFs')
        
        # plt.show()
        plt.tight_layout()
        plt.savefig(os.path.join(store_dirr, 'SF_of_neighboring_grains___WEIGHTED_average_SFs_of_neighbors_%d_FIPs' % num_to_consider))
        plt.close()  
        

        
        ''' Normalize by shared surface area and volume of neighbor '''
        
        num_to_consider = 50
        
        average_of_max_neighborhood_SFs = np.zeros((num_to_consider))
        average_of_max_SF_of_all_grains = np.zeros((num_to_consider))
        
        # average_of_max_SF_of_all_grains = np.max(All_grain_SFs[0:num_to_consider], axis = 1)
        
        for ppp in range(num_to_consider):
        
            ''' Do for the highest FIPs first '''
            # Let's examine the grains around the grain with the largest FIP
            largest_FIP_grain = int(all_data[ppp][1])
            nearest_neighbor_grains = [rr - 1 for rr in neighbor_list[largest_FIP_grain]]

            # CRUCIAL: GRAINS ARE ALSO ORIGINALLY INDEXED AT 1 !!!
            
            SFs_of_grains_neighboring_largest_FIP_grain = calculate_FCC_Schmid_Factors(orientations, nearest_neighbor_grains, [1,0,0])
        
            # Get largest Schmid Factor in the grains neighboring the largest FIP Grain
            # np.max(SFs_of_grains_neighboring_largest_FIP_grain, axis = 1)
            # np.mean(SFs_of_grains_neighboring_largest_FIP_grain, axis = 1)
        
            num_neigh_highest_FIP = len(SFs_of_grains_neighboring_largest_FIP_grain)
            
            # average_of_max_neighborhood_SFs[ppp] = np.mean(np.max(SFs_of_grains_neighboring_largest_FIP_grain, axis = 1))
            
            ''' Get shared area for this grain's neighbors: '''
            
            shared_area_btw_grain_and_neighbors = shared_surface_area_list[largest_FIP_grain]
            total_area = np.sum(shared_area_btw_grain_and_neighbors)
            
            neighbor_volumes = volumes[nearest_neighbor_grains]
            total_volume = np.sum(neighbor_volumes)
            
            normalized_SFs = np.zeros((num_neigh_highest_FIP,12))
            
            for ii in range(num_neigh_highest_FIP):
                normalized_SFs[ii] = (SFs_of_grains_neighboring_largest_FIP_grain[ii] * neighbor_volumes[ii] * shared_area_btw_grain_and_neighbors[ii]) / (total_volume*total_area)
            
            average_of_max_neighborhood_SFs[ppp] = np.mean(normalized_SFs)
            
            
            
            ''' Now repeat for the first 100 grains! '''
            
            nearest_neighbor_grains_first = [rr - 1 for rr in neighbor_list[ppp]]
            SFs_of_first_grains = calculate_FCC_Schmid_Factors(orientations, nearest_neighbor_grains_first, [1,0,0])
            num_neigh_highest_FIP_first_x_grains = len(SFs_of_first_grains)
            
            
            shared_area_btw_first_x_Grains = shared_surface_area_list[ppp]
            total_area_first_x_Grains = np.sum(shared_area_btw_first_x_Grains)
            
            
            neighbor_volumes_first_x_Grains = volumes[nearest_neighbor_grains_first]
            total_volume_first_x_Grains = np.sum(neighbor_volumes_first_x_Grains)
            
            
            normalized_SFs_first_x_grains = np.zeros((num_neigh_highest_FIP_first_x_grains,12))
        
            for ii in range(num_neigh_highest_FIP_first_x_grains):
                normalized_SFs_first_x_grains[ii] = (SFs_of_first_grains[ii] * neighbor_volumes_first_x_Grains[ii] * shared_area_btw_first_x_Grains[ii]) / (total_volume_first_x_Grains*total_area_first_x_Grains)
            
            average_of_max_SF_of_all_grains[ppp] = np.mean(normalized_SFs_first_x_grains)
            
            
        fig = plt.figure(facecolor="white", figsize=(7.5, 5), dpi=1200)     
        plt.plot(range(num_to_consider), SF_of_highest_FIPs[0:num_to_consider]/average_of_max_neighborhood_SFs, c = 'b', linestyle = ':',  label = '%d highest FIPs' % num_to_consider, zorder = 1) 
        plt.plot(range(num_to_consider), np.max(All_grain_SFs[0:num_to_consider],axis = 1)/average_of_max_SF_of_all_grains, c = 'r', linestyle = '-.', label = '%d first grains' % num_to_consider, zorder = 1) 
        
        # plt.scatter(range(num_to_consider), SF_of_highest_FIPs[0:num_to_consider]/average_of_max_neighborhood_SFs, c = 'b',  label = '%d highest FIPs' % num_to_consider, zorder = 1, s = 2) 
        # plt.scatter(range(num_to_consider), np.max(All_grain_SFs[0:num_to_consider],axis = 1)/average_of_max_SF_of_all_grains, c = 'r',  label = '%d first grains' % num_to_consider, zorder = 1, s = 2)   
        
        plt.legend(framealpha = 1)
        plt.grid('True', zorder = 2)
        # plt.ylim(0.9,1.25)
        
        plt.xlabel('Grain ID by FIP rank')
        plt.ylabel('Largest SF in grain / surface area weighted average of neighbors SFs')
        
        # plt.show()
        plt.tight_layout()
        plt.savefig(os.path.join(store_dirr, 'SF_of_neighboring_grains___WEIGHTED_average_SFs_of_neighbors_with_volume_%d_FIPs' % num_to_consider))
        plt.close()  



        ''' Expand to 2nd nearest neighbor grains ''' 
        # all_data variable is indexed at 0 !
        # neighbor_list variable is indexed at 1 so need to subtract 1 !

        num_to_consider = 50
        avg = True
        
        average_of_max_neighborhood_SFs = np.zeros((num_to_consider))
        average_of_max_SF_of_all_grains = np.zeros((num_to_consider))
        
        # average_of_max_SF_of_all_grains = np.max(All_grain_SFs[0:num_to_consider], axis = 1)
        
        for ppp in range(num_to_consider):
        
            ''' Do for the highest FIPs first '''
            # Let's examine the grains around the grain with the largest FIP
            largest_FIP_grain = int(all_data[ppp][1])
            nearest_neighbor_grains = [rr - 1 for rr in neighbor_list[largest_FIP_grain]]
            
            second_nearest_neighbor_grains = []
            
            for neighbors_1 in nearest_neighbor_grains:
                second_nearest_neighbor_grains.append(  [rr - 1 for rr in neighbor_list[neighbors_1]]   )

            flat_2nd_neighbors = list(flatten(second_nearest_neighbor_grains))
           

             
            # CRUCIAL: GRAINS ARE ALSO ORIGINALLY INDEXED AT 1 !!!
            
            SFs_of_grains_neighboring_largest_FIP_grain = calculate_FCC_Schmid_Factors(orientations, flat_2nd_neighbors, [1,0,0])
        
            # Get largest Schmid Factor in the grains neighboring the largest FIP Grain
            # np.max(SFs_of_grains_neighboring_largest_FIP_grain, axis = 1)
            # np.mean(SFs_of_grains_neighboring_largest_FIP_grain, axis = 1)
        
            num_neigh_highest_FIP = len(SFs_of_grains_neighboring_largest_FIP_grain)
            
            if not avg:
                average_of_max_neighborhood_SFs[ppp] = np.mean(np.max(SFs_of_grains_neighboring_largest_FIP_grain, axis = 1))
            else:
                average_of_max_neighborhood_SFs[ppp] = np.mean(SFs_of_grains_neighboring_largest_FIP_grain)
            
            print(num_neigh_highest_FIP)
            ''' Now repeat for the first 100 grains! '''
            
            nearest_neighbor_grains_first = [rr - 1 for rr in neighbor_list[ppp]]
            
            
            second_nearest_neighbor_grains_first_X_grains = []
            
            for neighbors_1 in nearest_neighbor_grains_first:
                second_nearest_neighbor_grains_first_X_grains.append(  [rr - 1 for rr in neighbor_list[neighbors_1]]   )

            flat_2nd_neighbors_first_x_Grains = list(flatten(second_nearest_neighbor_grains_first_X_grains))            

            
            SFs_of_first_grains = calculate_FCC_Schmid_Factors(orientations, flat_2nd_neighbors_first_x_Grains, [1,0,0])
            # np.max(SFs_of_first_grains, axis = 1)
            num_neigh_highest_FIP = len(SFs_of_first_grains)
        
            if not avg:
                average_of_max_SF_of_all_grains[ppp] = np.mean(np.max(SFs_of_first_grains, axis = 1))        
            else:
                average_of_max_SF_of_all_grains[ppp] = np.mean(SFs_of_first_grains)
            
            
        fig = plt.figure(facecolor="white", figsize=(7.5, 5), dpi=1200)     
        plt.plot(range(num_to_consider), SF_of_highest_FIPs[0:num_to_consider]/average_of_max_neighborhood_SFs, c = 'b', linestyle = ':',  label = '%d highest FIPs' % num_to_consider, zorder = 1) 
        plt.plot(range(num_to_consider), np.max(All_grain_SFs[0:num_to_consider],axis = 1)/average_of_max_SF_of_all_grains, c = 'r', linestyle = '-.', label = '%d first grains' % num_to_consider, zorder = 1) 
        
        # plt.scatter(range(num_to_consider), SF_of_highest_FIPs[0:num_to_consider]/average_of_max_neighborhood_SFs, c = 'b',  label = '%d highest FIPs' % num_to_consider, zorder = 1, s = 2) 
        # plt.scatter(range(num_to_consider), np.max(All_grain_SFs[0:num_to_consider],axis = 1)/average_of_max_SF_of_all_grains, c = 'r',  label = '%d first grains' % num_to_consider, zorder = 1, s = 2)   
        
        plt.legend(framealpha = 1)
        plt.grid('True', zorder = 2)
        # plt.ylim(0.9,1.25)
        
        plt.xlabel('Grain ID by FIP rank')
        if not avg:
            plt.ylabel('Largest SF in grain / average of max of neighbors SFs')
        else:
            plt.ylabel('Largest SF in grain / average of neighbors SFs')
        
        # plt.show()
        plt.tight_layout()
        if not avg:
            plt.savefig('SF_of_neighboring_grains___average_of_max_SFs_of_neighbors_%d_2nd_nearest.png' % num_to_consider)
        else:
            plt.savefig('SF_of_neighboring_grains___average_SFs_of_neighbors_%d_2nd_nearest.png' % num_to_consider)
        plt.close()           



        ''' Expand to 3rd nearest neighbor grains ''' 
        # all_data variable is indexed at 0 !
        # neighbor_list variable is indexed at 1 so need to subtract 1 !

        num_to_consider = 50
        avg = False
        
        average_of_max_neighborhood_SFs = np.zeros((num_to_consider))
        average_of_max_SF_of_all_grains = np.zeros((num_to_consider))
        
        # average_of_max_SF_of_all_grains = np.max(All_grain_SFs[0:num_to_consider], axis = 1)
        
        for ppp in range(num_to_consider):
        
            ''' Do for the highest FIPs first '''
            # Let's examine the grains around the grain with the largest FIP
            largest_FIP_grain = int(all_data[ppp][1])
            nearest_neighbor_grains = [rr - 1 for rr in neighbor_list[largest_FIP_grain]]
            
            second_nearest_neighbor_grains = []
            
            for neighbors_1 in nearest_neighbor_grains:
                second_nearest_neighbor_grains.append(  [rr - 1 for rr in neighbor_list[neighbors_1]]   )

            flat_2nd_neighbors = list(flatten(second_nearest_neighbor_grains))
            
            
            third_nearest_neighbor_grains = []
            
            for neighbors_1 in flat_2nd_neighbors:
                third_nearest_neighbor_grains.append(  [rr - 1 for rr in neighbor_list[neighbors_1]]   )

            flat_3rd_neighbors = list(flatten(third_nearest_neighbor_grains))            
            
            
            
             
            # CRUCIAL: GRAINS ARE ALSO ORIGINALLY INDEXED AT 1 !!!
            
            SFs_of_grains_neighboring_largest_FIP_grain = calculate_FCC_Schmid_Factors(orientations, flat_3rd_neighbors, [1,0,0])
        
            # Get largest Schmid Factor in the grains neighboring the largest FIP Grain
            # np.max(SFs_of_grains_neighboring_largest_FIP_grain, axis = 1)
            # np.mean(SFs_of_grains_neighboring_largest_FIP_grain, axis = 1)
        
            num_neigh_highest_FIP = len(SFs_of_grains_neighboring_largest_FIP_grain)
            
            if not avg:
                average_of_max_neighborhood_SFs[ppp] = np.mean(np.max(SFs_of_grains_neighboring_largest_FIP_grain, axis = 1))
            else:
                average_of_max_neighborhood_SFs[ppp] = np.mean(SFs_of_grains_neighboring_largest_FIP_grain)
            

            ''' Now repeat for the first 100 grains! '''
            
            nearest_neighbor_grains_first = [rr - 1 for rr in neighbor_list[ppp]]
            
            
            second_nearest_neighbor_grains_first_X_grains = []
            
            for neighbors_1 in nearest_neighbor_grains_first:
                second_nearest_neighbor_grains_first_X_grains.append(  [rr - 1 for rr in neighbor_list[neighbors_1]]   )

            flat_2nd_neighbors_first_x_Grains = list(flatten(second_nearest_neighbor_grains_first_X_grains))            
            
            
            third_nearest_neighbor_grains_first_X_grains = []
            
            for neighbors_1 in flat_2nd_neighbors_first_x_Grains:
                third_nearest_neighbor_grains_first_X_grains.append(  [rr - 1 for rr in neighbor_list[neighbors_1]]   )

            flat_3rd_neighbors_first_x_Grains = list(flatten(third_nearest_neighbor_grains_first_X_grains))                
            
            
            
            
            SFs_of_first_grains = calculate_FCC_Schmid_Factors(orientations, flat_3rd_neighbors_first_x_Grains, [1,0,0])
            # np.max(SFs_of_first_grains, axis = 1)
            num_neigh_highest_FIP = len(SFs_of_first_grains)
            

            
            if not avg:
                average_of_max_SF_of_all_grains[ppp] = np.mean(np.max(SFs_of_first_grains, axis = 1))        
            else:
                average_of_max_SF_of_all_grains[ppp] = np.mean(SFs_of_first_grains)
            
            
        fig = plt.figure(facecolor="white", figsize=(7.5, 5), dpi=1200)     
        plt.plot(range(num_to_consider), SF_of_highest_FIPs[0:num_to_consider]/average_of_max_neighborhood_SFs, c = 'b', linestyle = ':',  label = '%d highest FIPs' % num_to_consider, zorder = 1) 
        plt.plot(range(num_to_consider), np.max(All_grain_SFs[0:num_to_consider],axis = 1)/average_of_max_SF_of_all_grains, c = 'r', linestyle = '-.', label = '%d first grains' % num_to_consider, zorder = 1) 
        
        # plt.scatter(range(num_to_consider), SF_of_highest_FIPs[0:num_to_consider]/average_of_max_neighborhood_SFs, c = 'b',  label = '%d highest FIPs' % num_to_consider, zorder = 1, s = 2) 
        # plt.scatter(range(num_to_consider), np.max(All_grain_SFs[0:num_to_consider],axis = 1)/average_of_max_SF_of_all_grains, c = 'r',  label = '%d first grains' % num_to_consider, zorder = 1, s = 2)   
        
        plt.legend(framealpha = 1)
        plt.grid('True', zorder = 2)
        # plt.ylim(0.9,1.25)
        
        plt.xlabel('Grain ID by FIP rank')
        if not avg:
            plt.ylabel('Largest SF in grain / average of max of neighbors SFs')
        else:
            plt.ylabel('Largest SF in grain / average of neighbors SFs')
        
        # plt.show()
        plt.tight_layout()
        if not avg:
            plt.savefig(os.path.join(store_dirr, 'SF_of_neighboring_grains___average_of_max_SFs_of_neighbors_%d_3rd_nearest' % num_to_consider))
        else:
            plt.savefig(os.path.join(store_dirr, 'SF_of_neighboring_grains___average_SFs_of_neighbors_%d_3rd_nearest_%d_FIPs' % num_to_consider))
        plt.close()           


        
        # elements indexed at 0 as read in below!
        fname = 'element_grain_sets_0.p'
        h1 = open(os.path.join(directory, fname),'rb')
        grain_sets = p.load(h1)
        h1.close()        
     

        
        ''' 12 / 8 / 20: let's plot grain centroids and plastic shear strain range... '''
        grain_averaged_fips = np.zeros(len(grain_sets))
        max_PSS_per_elem = np.max(plastic_shear_strain_range_FIP, axis = 1)

        for ii, elems in enumerate(grain_sets):
            grain_averaged_fips[ii] = np.mean(max_PSS_per_elem[elems])   



        ''' Plot Schmid Factors ... '''
        for plot_me_plz in range(20):
            highest_fip_grain_index = plot_me_plz    
            #highest_fip_grain_index = 2
            
            highest_FIP_grain = int(all_data[highest_fip_grain_index][1])
            
            nearest_neighbor_grains = [rr - 1 for rr in neighbor_list[highest_FIP_grain]]
            
            x_cen = centroids[nearest_neighbor_grains].transpose()[0]
            y_cen = centroids[nearest_neighbor_grains].transpose()[1]
            z_cen = centroids[nearest_neighbor_grains].transpose()[2]
         
            # Get second nearest neighbors
            second_nearest_neighbor_grains = []
            for neighbors_1 in nearest_neighbor_grains:
                second_nearest_neighbor_grains.append(  [rr - 1 for rr in neighbor_list[neighbors_1]]   )
                
            flat_second = list(flatten(second_nearest_neighbor_grains))
            
            unique_second = []
            
            for item4 in flat_second:
                if item4 not in nearest_neighbor_grains:
                    unique_second.append(item4)
            
            unique_second.remove(highest_FIP_grain)
         
            x_cen_2nd = centroids[unique_second].transpose()[0]
            y_cen_2nd = centroids[unique_second].transpose()[1]
            z_cen_2nd = centroids[unique_second].transpose()[2]    
         
         
         
            All_grain_SFs = calculate_FCC_Schmid_Factors(orientations, [x for x in range(len(orientations))], [1,0,0])
         
            ''' f'''
            #     cvals  = [-2., -1, 2]
            #     colors = ["red","violet","blue"]

            #     norm=plt.Normalize(min(cvals),max(cvals))
            #     tuples = list(zip(map(norm,cvals), colors))
            #     cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)   
            
            
            fig = plt.figure(figsize=(16,10))
            ax = fig.add_subplot(111, projection='3d')
            
            cm = plt.cm.get_cmap('cool')
            
            p = ax.scatter(centroids[highest_FIP_grain][0], centroids[highest_FIP_grain][1], centroids[highest_FIP_grain][2], s = 200, c = [SF_of_highest_FIPs[highest_fip_grain_index]], marker = '^', label = 'Grain %d' % (int(all_data[highest_fip_grain_index][1]) + 1), cmap = cm, vmin = 0.4, vmax = 0.5)
            p = ax.scatter(x_cen, y_cen, z_cen, c = np.max(All_grain_SFs[nearest_neighbor_grains],axis = 1), marker = 'o', s = 200, label = '1st nearest neighbors', cmap = cm, vmin = 0.4, vmax = 0.5)
            # p = ax.scatter(x_cen_2nd, y_cen_2nd, z_cen_2nd, c = np.max(All_grain_SFs[unique_second],axis = 1), marker = 's', s = 200, label = '2nd nearest neighbors', cmap = cm)
            
            cbar = plt.colorbar(p)
            cbar.set_label("Max SF")
            
            # ax.scatter(layer_3_step_0_COM.transpose()[0], layer_3_step_0_COM.transpose()[1], layer_3_step_0_COM.transpose()[2], c='g', label = 'Layer 3')
            # ax.scatter(layer_4_step_0_COM.transpose()[0], layer_4_step_0_COM.transpose()[1], layer_4_step_0_COM.transpose()[2], c='r', label = 'Layer 4')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            plt.tight_layout()

            # plt.savefig('all_layer_3_grains.png')
            # plt.savefig('neighbors_surrounding_grain_index_%d_max_Schmid_factors_0.4_to_0.5.png' % plot_me_plz)
            plt.show()
            plt.close()   
            
            


        ''' Plot grain averaged plastic shear strain range '''
        
        for plot_me_plz in range(2):
            highest_fip_grain_index = plot_me_plz
            
            highest_FIP_grain = int(all_data[highest_fip_grain_index][1])
            
            nearest_neighbor_grains = [rr - 1 for rr in neighbor_list[highest_FIP_grain]]
            
            x_cen = centroids[nearest_neighbor_grains].transpose()[0]
            y_cen = centroids[nearest_neighbor_grains].transpose()[1]
            z_cen = centroids[nearest_neighbor_grains].transpose()[2]
         
            # Get second nearest neighbors
            second_nearest_neighbor_grains = []
            for neighbors_1 in nearest_neighbor_grains:
                second_nearest_neighbor_grains.append(  [rr - 1 for rr in neighbor_list[neighbors_1]]   )
                
            flat_second = list(flatten(second_nearest_neighbor_grains))
            
            unique_second = []
            
            for item4 in flat_second:
                if item4 not in nearest_neighbor_grains:
                    unique_second.append(item4)
            
            unique_second.remove(highest_FIP_grain)
         
            x_cen_2nd = centroids[unique_second].transpose()[0]
            y_cen_2nd = centroids[unique_second].transpose()[1]
            z_cen_2nd = centroids[unique_second].transpose()[2]    
         
         
         
            All_grain_SFs = calculate_FCC_Schmid_Factors(orientations, [x for x in range(len(orientations))], [1,0,0])
         
            ''' f'''
            #     cvals  = [-2., -1, 2]
            #     colors = ["red","violet","blue"]

            #     norm=plt.Normalize(min(cvals),max(cvals))
            #     tuples = list(zip(map(norm,cvals), colors))
            #     cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)   
            
            
            fig = plt.figure(figsize=(16,10))
            ax = fig.add_subplot(111, projection='3d')
            
            cm = plt.cm.get_cmap('cool')
            
            p = ax.scatter(centroids[highest_FIP_grain][0], centroids[highest_FIP_grain][1], centroids[highest_FIP_grain][2], s = 200, c = [grain_averaged_fips[int(all_data[highest_fip_grain_index][1])]], marker = '^', label = 'Grain %d' % (int(all_data[highest_fip_grain_index][1]) + 1), cmap = cm, vmin=np.min(grain_averaged_fips[nearest_neighbor_grains]), vmax=grain_averaged_fips[int(all_data[highest_fip_grain_index][1])])
            p = ax.scatter(x_cen, y_cen, z_cen, c = grain_averaged_fips[nearest_neighbor_grains], marker = 'o', s = 200, label = '1st nearest neighbors', cmap = cm, vmin=np.min(grain_averaged_fips[nearest_neighbor_grains]), vmax=grain_averaged_fips[int(all_data[highest_fip_grain_index][1])])
            # p = ax.scatter(x_cen_2nd, y_cen_2nd, z_cen_2nd, c = np.max(All_grain_SFs[unique_second],axis = 1), marker = 's', s = 200, label = '2nd nearest neighbors', cmap = cm)
            
            cbar = plt.colorbar(p)
            cbar.set_label("Grain averaged maximum plastic shear strain range")
            
            # ax.scatter(layer_3_step_0_COM.transpose()[0], layer_3_step_0_COM.transpose()[1], layer_3_step_0_COM.transpose()[2], c='g', label = 'Layer 3')
            # ax.scatter(layer_4_step_0_COM.transpose()[0], layer_4_step_0_COM.transpose()[1], layer_4_step_0_COM.transpose()[2], c='r', label = 'Layer 4')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            plt.tight_layout()

            # plt.savefig('neighbors_surrounding_grain_index_%d.png' % plot_me_plz)
            plt.show()
            plt.close()   
        

        # Other quick calculations
        # "Lowest" possible SFs when pulling along the [111] direction
        # calculate_FCC_Schmid_Factors(np.zeros((2,3)),[0],[1,1,1]/np.sqrt(3))
        
        # straing along [100] direction:
        # calculate_FCC_Schmid_Factors(np.zeros((2,3)),[0],[1,0,0])
    
    
def main():
    # Plot correlations
    compute_correlations()
    
    # Visualize the top 50 FIP grains
    top_50_FIPs_vtk()

if __name__ == "__main__":
    main()
   
