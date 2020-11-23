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

# Get name of directory that contains the PRISMS-Fatigue scripts
DIR_LOC = os.path.dirname(os.path.abspath(__file__))

locats_Al7075_compare_BC_prisms =   [r'C:\Users\stopk\Documents\Research\PRISMS\May_large_scale_comparison\cubic_equiaxed_periodic',
                                     r'C:\Users\stopk\Documents\Research\PRISMS\May_large_scale_comparison\cubic_equiaxed_free_surface',
                                     r'C:\Users\stopk\Documents\Research\PRISMS\May_large_scale_comparison\random_equiaxed_periodic',
                                     r'C:\Users\stopk\Documents\Research\PRISMS\May_large_scale_comparison\random_equiaxed_free_surface',
                                     r'C:\Users\stopk\Documents\Research\PRISMS\May_large_scale_comparison\rolled_equiaxed_periodic',
                                     r'C:\Users\stopk\Documents\Research\PRISMS\May_large_scale_comparison\rolled_equiaxed_free_surface']    


locats_Al7075_compare_shape_prisms = [r'C:\Users\stopk\Documents\Research\PRISMS\May_large_scale_comparison\cubic_equiaxed_periodic',
                                      r'C:\Users\stopk\Documents\Research\PRISMS\May_large_scale_comparison\cubic_elongated_periodic',
                                      r'C:\Users\stopk\Documents\Research\PRISMS\May_large_scale_comparison\random_equiaxed_periodic',
                                      r'C:\Users\stopk\Documents\Research\PRISMS\May_large_scale_comparison\random_elongated_periodic',
                                      r'C:\Users\stopk\Documents\Research\PRISMS\May_large_scale_comparison\rolled_equiaxed_periodic',
                                      r'C:\Users\stopk\Documents\Research\PRISMS\May_large_scale_comparison\rolled_elongated_periodic']  



locats_PRISMS_Fatigue_demo = [os.path.dirname(DIR_LOC) + '\\PRISMS-Fatigue_tutorial\\cubic_tutorial',
                              os.path.dirname(DIR_LOC) + '\\PRISMS-Fatigue_tutorial\\random_tutorial']


pad_dist = 12
font_size = 8
tick_widths = 1
tick_lens = 5


def read_FIPs_from_single_SVE():
    # Additional unused function to read in the largest sub-band averaged FIPs (one per grain) from a single microstructure instantiation
    # This is particularly useful when considering a very large SVE with more than ~10,000 grains as this function can take a long time!

    # Specify name of pickle file with sub-band averaged FIPs
    fname = 'sub_band_averaged_FS_FIP_pickle_0.p'    
    
    # Specify how many of the highest FIPs per grain should be imported. Typically, only the few hundred highest FIPs are of interest
    # This significantly speeds up this algorithm!
    # IMPORTANT: If this is set below the number of grains in the instantiation, the function will fail! 
    get_num_FIPs = 250
    
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
            print(mm)
        nn += 1     
    
    return new_all_fs_fips, all_data, added_g 

def read_pickled_SBA_FIPs_with_location(directory, num_fips, FIP_type, averaging_type, non_periodic_dir):
    # Inputs:
    #    Directory         : Location of batch folder
    #    num_fips          : Number of FIPs to plot from the entire batch folder
    #    non_periodic_dir  : Non-periodic direction

    # Read in SBA FIPs from one batch folder
    # NOTE: Read in Top num_fips from each batch folder
    # Additionally, read in location of the sub-band region with maximum SBA FIP
    
    # Go to directory
    tmp_dir = os.getcwd()
    os.chdir(directory)

    # Find all pickle files with FIPs
    file_names = []
    
    # Specify type of desired FIP averaging to plot
    if averaging_type == 'sub_band':
        # fname = 'prisms_sub_band_averaged_FIPs_form_pickled*'
        fname = 'sub_band_averaged_%s_pickle*' % FIP_type
        
    # elif averaging_type == 'band':
    #     fname = 'band_averaged_%s_pickle*' % FIP_type
    # elif averaging_type == 'grain':
    #     fname = 'grain_averaged_%s_pickle*' % FIP_type
        
    else:
        # THIS FUNCTION ONLY WORKS WITH THE SUB-BAND AVERAGED FIPS BECAUSE IT LOCATES THE LOCATION OF SUB-BANDS SPECIFICALLY
        raise ValueError('This function is only compatible with the "sub_band" averaged FIPS!')
    
    for Name in glob.glob(fname):
        file_names.append(Name)
    
    inst_num = []
    
    for kk in file_names:
        inst_num.append(int(re.search(r'\d+', kk).group()))

    # Make sure the desired files exist in this folder, i.e., that the FIPs have been volume averaged as desired
    if len(file_names) == 0:
        raise ValueError('No FIP files detected! Please double check settings!')
    
    print('Currently in %s' % directory)
    
    silly_count = 0
    new_all_fs_fips = []
    
    
    # Read in element centroids
    # These should technically be the same for all instantiations
    
    fname2 = os.path.join(os.getcwd(), 'El_pos_0.p')
    h2 = open(fname2,'rb')
    el_cen = p.load(h2)
    h2.close()    
    
    for kk in inst_num:
    
        added_g = []
        
        # Read in FIPs
        
        # fname1 = os.path.join(os.getcwd(), 'prisms_sub_band_averaged_FIPs_form_pickled_%d.p' % kk)
        # print(kk)

        if averaging_type == 'sub_band':
            fname1 = 'sub_band_averaged_%s_pickle_%d.p' % (FIP_type, kk)
        # elif averaging_type == 'band':
        #     fname1 = 'band_averaged_%s_pickle_%d.p' % (FIP_type, kk)
        # elif averaging_type == 'grain':
        #     fname1 = 'grain_averaged_%s_pickle_%d.p' % (FIP_type, kk)
        else:
            # THIS FUNCTION ONLY WORKS WITH THE SUB-BAND AVERAGED FIPS BECAUSE IT LOCATES THE LOCATION OF SUB-BANDS SPECIFICALLY
            raise ValueError('This function is only compatible with the "sub_band" averaged FIPS!')        
        print(kk)
        
            
        h1 = open(fname1,'rb')
        fips = p.load(h1)
        h1.close()
        sorted_fips = sorted(fips.items(), key=operator.itemgetter(1))
        sorted_fips.reverse()
        
        # Read in sub band information
        
        fname1 = os.path.join(os.getcwd(), 'sub_band_info_%d.p' % kk)
        h2 = open(fname1,'rb')
        master_sub_band_dictionary,number_of_layers,number_of_sub_bands = p.load(h2, encoding = 'latin1')
        h2.close()        


        # Iterate through all FIPs, extract the maximum SBA FIP per grain
    
        for nn in range(len(sorted_fips)):
            if sorted_fips[nn][0][0] not in added_g:
            
                added_g.append(sorted_fips[nn][0][0])
               
                SBA_FIP = sorted_fips[nn][1]
                
                # Find sub-band for this SBA FIP
                grain_number = sorted_fips[nn][0][0]
                slip_system_number = sorted_fips[nn][0][1]
                layer_number = sorted_fips[nn][0][2]
                sub_band_number = sorted_fips[nn][0][3]

                # NEED TO SUBTRACT 1 FOR EVERY ELEMENT NUMBER TO INDEX AT 0 WHEN CALCULATING SUB BAND CENTROID !!

                
                # ********* THIS SECTION OF THE CODE IS ELEMENT SPECIFIC DUE TO THE DIFFERENT ARRANGEMENT IN SLIP PLANES / BANDING PROCEDURE !! *********
                # Find elements in this sub-band
                # Divide slip_system_number by 3 to get the correct set of planes! (0,0,0,1,1,1,2,2,2,3,3,3) from (0,1,2,3,4,5,6,7,8,9,10,11)
                elems = master_sub_band_dictionary[grain_number,int(slip_system_number/3),layer_number,sub_band_number]
                elems = [x - 1 for x in elems]
               
                ### FOR SIMULATIONS NON-PERIODIC IN THE X DIRECTION
                # All are named average_y_location for simplicity
                if non_periodic_dir == 'X':
                    # Calculate centroid of sub band in X direction
                    average_y_location = np.mean(el_cen[elems].transpose()[0])  
                    
                ### FOR SIMULATIONS NON-PERIODIC IN THE Y DIRECTION                    
                elif non_periodic_dir == 'Y':
                    # Calculate centroid of sub band in Y direction
                    average_y_location = np.mean(el_cen[elems].transpose()[1])     
                    
                ### FOR SIMULATIONS NON-PERIODIC IN THE Z DIRECTION
                elif non_periodic_dir == 'Z':
                    # Calculate centroid of sub band in Z direction
                    average_y_location = np.mean(el_cen[elems].transpose()[2])

                # Store into array
                if silly_count == 0:
                    fip_loc_array = np.array([[SBA_FIP,average_y_location]])
                    silly_count += 1
                else:
                    temp23 = np.array([[SBA_FIP,average_y_location]])
                    fip_loc_array = np.append(fip_loc_array,temp23,axis=0)
                    
    sorted_fip_loc_array = fip_loc_array[fip_loc_array[:,0].argsort()][::-1]            
    os.chdir(tmp_dir)
    # Return array of FIPs and corresponding SBA centroids
    return sorted_fip_loc_array[0:num_fips]

def read_pickled_SBA_FIPs(directory, num_fips_extract, FIP_type, averaging_type):
    # Read in FIPs from one batch folder
    
    # Go to directory
    tmp_dir = os.getcwd()
    os.chdir(directory)
    
    # Find all pickle files with FIPs
    file_names = []
    
    # Specify type of desired FIP averaging to plot
    if averaging_type == 'sub_band':
        # fname = 'prisms_sub_band_averaged_FIPs_form_pickled*'
        fname = 'sub_band_averaged_%s_pickle*' % FIP_type
    elif averaging_type == 'band':
        fname = 'band_averaged_%s_pickle*' % FIP_type
    elif averaging_type == 'grain':
        fname = 'grain_averaged_%s_pickle*' % FIP_type
    else:
        raise ValueError('Unknown input! Please ensure averaging_type is "sub_band", "band", or "grain"!')
    
    for Name in glob.glob(fname):
        file_names.append(Name)
       
    # Make sure the desired files exist in this folder, i.e., that the FIPs have been volume averaged as desired
    if len(file_names) == 0:
        raise ValueError('No FIP files detected! Please double check settings!')
    
    print('Currently in %s' % directory)  
    
    # Initialize list to store FIPs
    new_all_fs_fips = []
    
    # Sub band and band averaged FIPs are extracted in the same fashion
    if averaging_type == 'sub_band' or averaging_type == 'band':
    
        # Iterate through all pickle FIP files
        for fip_file in file_names:
        
            # Extract the single highest FIP per grain
            added_g = []
            
            # Read in FIPs
            fname1 = os.path.join(os.getcwd(), fip_file)
            h1 = open(fname1,'rb')
            fips = p.load(h1)
            h1.close()
            print(fip_file)
            
            sorted_fips = sorted(fips.items(), key=operator.itemgetter(1))
            sorted_fips.reverse()
            
            # Iterate through all FIPs, extract the maximum SBA FIP per grain
            for nn in range(len(sorted_fips)):
                if sorted_fips[nn][0][0] not in added_g:
                    added_g.append(sorted_fips[nn][0][0])
                    new_all_fs_fips.append(sorted_fips[nn][1])

        new_all_fs_fips.sort(reverse=True)
        os.chdir(tmp_dir)    
        return new_all_fs_fips[0:num_fips_extract]
        
    elif averaging_type == 'grain':    
        
        new_all_fs_fips = np.asarray(())
        
        # Iterate through all pickle FIP files
        for fip_file in file_names:
        
            # Read in FIPs
            fname1 = os.path.join(os.getcwd(), fip_file)
            h1 = open(fname1,'rb')
            fips = p.load(h1)
            h1.close()
            print(fip_file)
            
            # Append entire list of FIPs to the main list since these are already the largest FIPs per grain
            new_all_fs_fips = np.append(new_all_fs_fips, fips)
         
        all_sorted_grain_FIPs = np.sort(new_all_fs_fips)[::-1]
        os.chdir(tmp_dir)    
        return all_sorted_grain_FIPs[0:num_fips_extract]    

def plot_EVD_FIP_distance(base_directory, mat, num_fips_plot, FIP_type, averaging_type, non_periodic_dir, SVE_non_periodic_length):
    # Function to plot sub band averaged FIPs and their proximity to the free surface

    # Change directory to store plots and compiled FIPs in new folder
    os.chdir(base_directory)

    # Number of total FIPs to extract from simulation files
    num_fips_extract = 500
    
    if num_fips_extract < num_fips_plot:
        raise ValueError('More FIPs are desired for plots than are to be extracted! Please edit "num_fips_plot" and/or "num_fips_extract" !')
    
    print('Plotting SBA FIPs vs distance to free surface')
    
    
    if mat == 'prisms_compare_bcs':
        
        # Names to use for figure legend
        names = ["Cubic Periodic", "Cubic Free Surface", "Random Periodic", "Random Free Surface", "Rolled Periodic", "Rolled Free Surface"]
        
        # Locations of actual FIP files
        locs = locats_Al7075_compare_BC_prisms
        
        # Number of columns for figure legend
        plot_col = 3
        
        # Marker colors
        cfm = ['r','r','b','b','g','g']
        
        # Marker shapes
        sfm = ['o','o','s','s','^','^']   
        
        # If True, every other FIP dataset will be plotted with hollow markers
        hollow_and_full = True  
        
    elif mat == 'PRISMS_Fatigue_demo':
        # Names to use for figure legend
        names = ["Cubic", "Random"]
        
        # Locations of actual FIP files
        locs = locats_PRISMS_Fatigue_demo
        
        # Number of columns for figure legend
        plot_col = 1
        
        # Marker colors
        cfm = ['r','b']
        
        # Marker shapes
        sfm = ['o','s']   
        
        # If True, every other FIP dataset will be plotted with hollow markers
        hollow_and_full = False
    
    # Initialize list of FIPs and distances
    fip_and_loc = []    
    
    for pickle_location in locs:
        file_name_1 = os.path.basename(os.path.split(pickle_location)[1])
        compiled_fips_loc = os.path.join(os.getcwd(), 'compiled_surf_dist_%s_%s_%s.p' % (FIP_type, averaging_type, file_name_1))
        
        # If pickle file of compiled FIPs and distances already exists, just read it in
        if os.path.exists(compiled_fips_loc):
            h1 = open(compiled_fips_loc,'rb')
            fip_and_loc.append(p.load(h1))
            h1.close()    
        else:
            
            # Otherwise, read and compile FIPs and distances for all microstructures into a single pickle file
            temp_fips_5 = read_pickled_SBA_FIPs_with_location(pickle_location, num_fips_extract, FIP_type, averaging_type, non_periodic_dir)
            fip_and_loc.append(temp_fips_5)
            
            h4 = open(compiled_fips_loc,'wb')
            p.dump(temp_fips_5,h4)
            h4.close()    


    fig = plt.figure(facecolor="white", figsize=(6, 4), dpi=1200)
    avg_locs = []
    
    # Store average distance to the free surface in text file
    ftext = os.path.join(os.getcwd(),'Avg_surf_dist_for_top_%d_%s.txt' % (num_fips_plot,mat))
    h6 = open(ftext, 'w')
    
    y_plot_values = []
    
    # Assign the correct SVE size (half length) in mm for plotting purposes
    sve_half_length = SVE_non_periodic_length / 2

    for i, d in enumerate(fip_and_loc):
        
        x = d.transpose()[1]
        x_calced = [sve_half_length - abs(item - sve_half_length) for item in x][0:num_fips_plot]
        # Change to micrometers from millimeters
        x_calced = [iii * 1000 for iii in x_calced]
        
        y = d.transpose()[0][0:num_fips_plot]
        
        # Plot alternating hollow and full markers to easily distinguish between surface and bulk FIPs
        if hollow_and_full:
            if i % 2 == 0:
                plt.scatter(x_calced, y, c=cfm[i], label=names[i], marker=sfm[i], alpha=0.75, linewidths=0.5, zorder = 2)
            
            elif i % 2 == 1:
                plt.scatter(x_calced, y, c='white', edgecolors=cfm[i], label=names[i], marker=sfm[i], alpha=0.75, linewidths=1, zorder = 2)            
        
        # Otherwise, plot all markers with solid colors
        else:
            plt.scatter(x_calced, y, c=cfm[i], label=names[i], marker=sfm[i], alpha=0.75, linewidths=0.5, zorder = 2)
                
                
        print(np.mean(x_calced))
        avg_locs.append(np.mean(x_calced))
        h6.write('%0.5f\n' % np.mean(x_calced))
        
        y_plot_values.append(y)
    
    y_min = np.min(y_plot_values)
    y_max = np.max(y_plot_values)
    
    h6.close()
    
    # Other figure formatting commands
    plt.ylabel("FIP", fontsize='medium')
    plt.xlabel('Distance from the free surface ['+u"\u03bcm]", fontsize='medium')
    plt.grid(True, zorder = 1)
    plt.xlim(-0.0025, 40)
    # plt.ylim(y_min - y_min  * 0.2, y_max * 1.2)
    plt.legend(loc='upper right', fontsize='8', ncol=plot_col, framealpha=1)
    plt.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0), useMathText = True)
    plt.tight_layout()
    plt.savefig("SBA_FIP_surface_dist_%d_%s" % (num_fips_plot,mat), dpi=fig.dpi)
    
def plot_FIPS(base_directory, plt_type, mat, num_fips_plot, FIP_type, averaging_type, save_fig = True):
    # Function to plot FIPs as Extreme Value Distributions (EVDs)
    
    print('Plotting volume averaged FIPs')
    os.chdir(base_directory)

    # Specify which FIPS to plot
    
    # This will plot FIPs from subsurface/bulk and free-surface simulations
    if mat == 'prisms_compare_bcs':
        # Names to use for figure legend
        # These must match the order of simulation folders specified at the top of this script
        names = ["Cubic Periodic","Cubic Free Surface","Random Periodic","Random Free Surface","Rolled Periodic","Rolled Free Surface"]
        
        # Locations of actual FIP files
        locs = locats_Al7075_compare_BC_prisms
        
        # Number of columns for figure legend
        plot_col = 1
        
        # Marker colors
        cfm = ['r','r','b','b','g','g']
        
        # Marker shapes
        sfm = ['o','o','s','s','^','^']
        
        # If True, FIPs from periodic and free surface simulations will be hollow and full, respectively
        hollow_and_full = True   

    # This will plot FIPs from fully periodic simulations with different crystallographic textures and grain morphologies
    elif mat == 'prisms_compare_shape':
        # Names to use for figure legend
        names = ["Cubic Equiaxed","Cubic Elongated","Random Equiaxed","Random Elongated","Rolled Equiaxed","Rolled Elongated"]
        
        # Locations of actual FIP files
        locs = locats_Al7075_compare_shape_prisms
        
        # Number of columns for figure legend
        plot_col = 1
        
        # Marker colors
        cfm = ['r','r','b','b','g','g']
        
        # Marker shapes
        sfm = ['o','o','s','s','^','^']   
        
        # If True, every other FIP dataset will be plotted with hollow markers
        hollow_and_full = True
        
    # This will plot FIPs from the PRISMS-Fatigue Tutorial
    elif mat == 'PRISMS_Fatigue_demo':
        # Names to use for figure legend
        names = ["Cubic", "Random"]
        
        # Locations of actual FIP files
        locs = locats_PRISMS_Fatigue_demo
        
        # Number of columns for figure legend
        plot_col = 1
        
        # Marker colors
        cfm = ['r','b']
        
        # Marker shapes
        sfm = ['o','s']   
        
        # If True, every other FIP dataset will be plotted with hollow markers
        hollow_and_full = False

    else:
        raise ValueError('Please check "mat" variable!')

 
    # Number of total FIPs to extract from simulation files
    num_fips_extract = 500
    
    if num_fips_extract < num_fips_plot:
        raise ValueError('More FIPs are desired for plots than are to be extracted! Please edit "num_fips_plot" and/or "num_fips_extract" !')
   

    # If compiled list of FIPs exists, simply read it in; otherwise, compile FIPs
    fip_list = []
    for pickle_location in locs:
    
        # Define name of file with compiled FIPs for each simulation folder
        file_name_1 = os.path.basename(os.path.split(pickle_location)[1])
        compiled_fips_loc = os.path.join(os.getcwd(), 'compiled_%s_%s_%s.p' % (FIP_type, averaging_type, file_name_1))

        # If this file already exists, read in the FIPs
        if os.path.exists(compiled_fips_loc):
            h1 = open(compiled_fips_loc,'rb')
            fip_list.append(p.load(h1, encoding = 'latin1'))
            h1.close()

        # Otherwise, read and store FIPs for quicker plotting
        else:

            temp_fs_fips = read_pickled_SBA_FIPs(pickle_location, num_fips_extract, FIP_type, averaging_type)
            print('Reading SBA surface and subsurface FIPs')
            temp_fs_fips = np.array(temp_fs_fips)

            if len(temp_fs_fips) > 0:
                fip_list.append(temp_fs_fips)
                
            h2 = open(compiled_fips_loc,'wb')
            p.dump(temp_fs_fips,h2)
            h2.close()


    # Create list of appended FIPs to plot
    fip_list_to_plot = []
    for item in fip_list:
        fip_list_to_plot.append(item[0:num_fips_plot])
    
    # Print the number of FIPs to be plotted
    for item in fip_list_to_plot:
        # print len(item)
        print('Length of FIP list: ' + str(len(item)))


    # Option to plot either Frechet or Gumbel distribution with band-averaged FIPs
    if plt_type == 'frechet':
        fig, r_squared_values, slope_values, y_intercept_values, highest_FIP = plot_gumbels(fip_list_to_plot, plt_type, cfm, sfm, mat, plot_col, names, hollow_and_full, xlabel = "ln(FS FIPs)")
    elif plt_type == 'gumbel':
        fig, r_squared_values, slope_values, y_intercept_values, highest_FIP = plot_gumbels(fip_list_to_plot, plt_type, cfm, sfm, mat, plot_col, names, hollow_and_full, xlabel = "FIP")        
    
    
    if save_fig:
        plt.savefig("%s_EVD_%s_%s_%d_%s" % (FIP_type, averaging_type, plt_type, num_fips_plot, mat), dpi=fig.dpi)
        
    # Write r-squared values to .csv file 
    fname = os.path.join(os.getcwd(),'r_squared_%s_%s_%s_%d_%s.csv' % (plt_type, FIP_type, averaging_type, num_fips_plot, mat))    
    write_r = open(fname,'w')
    for i in range(len(r_squared_values)):
        write_r.write(str(r_squared_values[i]) + ',' + str(slope_values[i]) + ',' + str(y_intercept_values[i])+ ',' + str(highest_FIP[i]) + '\n')
    write_r.close()
    print('SBA FIP r-squared value for ' + plt_type + ' is ' + str(np.average(r_squared_values)) + ' for ' + str(num_fips_plot) + ' FIPs') 
   
def prettify_frame(ax):

    # As the name implies, this function simply makes the plot pretty :) 
    
    ax.tick_params(which='both', direction='out', pad=pad_dist*.75)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(tick_widths)
        ax.spines[axis].set_visible(True)
        ax.spines[axis].set_color('k')
    ax.xaxis.set_tick_params(which='major', width=tick_widths,length=tick_lens,color='k')
    ax.xaxis.set_tick_params(which='minor', width=tick_widths/2.0,length=tick_lens*.6,color='k')
    ax.yaxis.set_tick_params(which='major', width=tick_widths,length=tick_lens,color='k')
    ax.yaxis.set_tick_params(which='minor', width=tick_widths/2.0,length=tick_lens*.6,color='k')
    temp_list = [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    for item in temp_list:
        item.set_fontsize(font_size * 1.25)
        item.set_color("k")
    try:
        for item in ax.get_legend().get_texts():
            item.set_fontsize(font_size * 1)
            item.set_color("k")
    except:
        print("not doing legend text...")
    ax.set_frame_on(True)
    plt.tight_layout()
    
def plot_gumbels(data, plt_type, cfm, sfm, mat, plot_col, names = None, hollow_and_full = False, xlabel = "", plot_lin = False):
    
    # Fit FIPs to EVD
    
    if names is None:
        names = [str(i) for i in range(len(data))]

    # Plot all FIPs for comparison
    fig = plt.figure(facecolor="white", figsize=(6, 4), dpi=1200)
    ax = plt.axes()
    plt.grid(linewidth = 0.5)
    ax.set_axisbelow(True)
    d_min = None
    d_max = None
    min_ys = 0
    max_ys = 0
    
    r_squared_values     = []
    slope_values         = []
    y_intercept_values   = []
    highest_FIP          = []
    
    for i, d_orig in enumerate(data):

        d = np.sort(d_orig)
        
        # If FIPs are to be fit to the Frechet EVD, take the log of each FIP
        if plt_type == 'frechet':
            for yyy in range(len(d)):
                d[yyy] = np.log(d[yyy])
        
        # Estimate probability; see references below that explain this equation
        ys = (np.arange(len(d)) + 1 - 0.3) / (len(d) + 0.4)
        ys = -np.log(-np.log(ys))
        
        min_ys = min(min_ys, ys[0])
        max_ys = max(max_ys, ys[-1])
        colormat = cm.rainbow(np.linspace(0, 1, 4))

        # If true, plot every other set of FIPs as hollow for easy visual comparison
        if hollow_and_full:
            if i % 2 == 0:
                ax.scatter(d, ys, c=cfm[i], label=names[i], marker=sfm[i], alpha=0.75, linewidths=0.5)
            
            elif i % 2 == 1:
                ax.scatter(d, ys, c='white', edgecolors=cfm[i], label=names[i], marker=sfm[i], alpha=0.75, linewidths=1)       
        
        # Otherwise, plot all markers as solid color
        else:
            ax.scatter(d, ys, c=cfm[i], label=names[i], marker=sfm[i], alpha=0.75, linewidths=0.5)
           
        # If plot_lin is true, a line of best fit is also plotted
        slope, intercept, r_value, p_value, std_err = ss.linregress(np.ndarray.tolist(d),np.ndarray.tolist(ys))
        interp_points = np.asarray([np.min(d), np.max(d)])
        interp_ys = interp_points * slope + intercept
        if plot_lin:
            ax.plot(interp_points, interp_ys, c='k', linewidth=1, linestyle='--')
        # print(len(d))
        print("slope: %e, intercept: %e, r^2: %f" % (slope, intercept, r_value**2))
        
        if d_max is None or np.max(d) > d_max:
            d_max = np.max(d)
        if d_min is None or np.min(d) < d_min:
            d_min = np.min(d)
        r_squared_values.append(r_value**2)
        slope_values.append(slope)
        highest_FIP.append(max(d_orig))
        y_intercept_values.append(intercept)

    
    plt.ylabel("$-\mathrm{ln}(-\mathrm{ln}(p))$", fontsize='8')
    plt.xlabel(xlabel, fontsize='8')
    plt.legend(loc='best', fontsize='8', ncol=plot_col, framealpha=1)

    # Specify upper limit on plot; edit this as necessary 
    if len(d) == 100:
        plt.ylim(np.floor(min_ys), 6)
    elif len(d) == 50:
        plt.ylim(np.floor(min_ys), 5)
        
    if plt_type == 'gumbel':
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText = True)
    elif plt_type == 'frechet':
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText = True)
        
    prettify_frame(ax)
    plt.tight_layout()
    return fig, r_squared_values, slope_values, y_intercept_values, highest_FIP
    
def main():
    # Specify type of analysis: either 'fip_evd' to plot FIP extreme value distributions (EVDs) or 'surf_dist' to plot the highest FIPs as a function of their distance to the free surface 
    analysis_type = 'fip_evd'
    
    # Specify folder which contains all of the simulation batch folders
    # IMPORTANT: this directory should contain the locations specified at the top of this python script, NOT the individual folders with the instantiations!
    # This script goes through EACH ONE of the folders specified and extracts the relevant information for plotting purposes
    
    directory = os.path.dirname(DIR_LOC) + '\\tutorial\\test_run_1'
    
    # directory = r'C:\Users\stopk\Documents\GitHub\PRISMS-Fatigue\tutorial\test_run_1'

    # Specify which 'material' to plot, i.e., which combination of microstructure folders
    # Please see the top of the "plot_FIPS" function
    mat = 'PRISMS_Fatigue_demo'
    
    # Specify which FIP to import
    FIP_type = 'FS_FIP'
    
    # Specify which volume averaging scheme to import. By default, the sub-band averaged (SBA) FIPs are used
    # IMPORTANT: this will fail if the FIPs have not yet been averaged in the desired manner (see 'volume_average_FIPs.py' script)
    # Options: 'sub_band', 'band', and 'grain'
    averaging_type = 'sub_band'
    
    if analysis_type == 'fip_evd':
    
        # Specify how many of the highest FIPs should be plotted
        n_fip_plot = 50
    
        # Specify whether FIPs should be fit to the "gumbel" or "frechet" EVD
        plt_type = 'gumbel'
        
        # Call function to plot FIPs
        plot_FIPS(directory, plt_type, mat, n_fip_plot, FIP_type, averaging_type, save_fig = True)
        
    elif analysis_type == 'surf_dist':
        # Plot SBA FIP vs distance from free surface:
        
        # Specify how many of the highest FIPs should be plotted
        n_fip_plot = 10 
        
        # Specify the non-periodic direction (i.e., the set of parallel faces set to traction-free conditions)
        # In the references associated with these scripts (see below), this is set to the 'Y' direction
        non_periodic_dir = 'Y'
        
        # Specify the size/length of the SVE in the non-periodic direction in mm, to properly locate the FIPs within the microstructure 
        SVE_non_periodic_length = 0.0725
        
        # Call function to plot FIPs
        plot_EVD_FIP_distance(directory, mat, n_fip_plot, FIP_type, averaging_type, non_periodic_dir, SVE_non_periodic_length)

if __name__ == "__main__":
    main()

# References with more information on these types of simulations:
# Stopka, K.S., McDowell, D.L. Microstructure-Sensitive Computational Estimates of Driving Forces for Surface Versus Subsurface Fatigue Crack Formation in Duplex Ti-6Al-4V and Al 7075-T6. JOM 72, 28–38 (2020). https://doi.org/10.1007/s11837-019-03804-1

# Stopka and McDowell, “Microstructure-Sensitive Computational Multiaxial Fatigue of Al 7075-T6 and Duplex Ti-6Al-4V,” International Journal of Fatigue, 133 (2020) 105460.  https://doi.org/10.1016/j.ijfatigue.2019.105460

    

    
    