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

# Get name of directory that contains the PRISMS-Fatigue scripts
DIR_LOC = os.path.dirname(os.path.abspath(__file__))

locats_Al7075_compare_shape_7500_grain = [os.path.join(DIR_LOC, r'Section_3\7500_grain\7500_grain_cubic_equiaxed'),
                                          os.path.join(DIR_LOC, r'Section_3\7500_grain\7500_grain_cubic_elongated'),
                                          os.path.join(DIR_LOC, r'Section_3\7500_grain\7500_grain_random_equiaxed'),
                                          os.path.join(DIR_LOC, r'Section_3\7500_grain\7500_grain_random_elongated'),
                                          os.path.join(DIR_LOC, r'Section_3\7500_grain\7500_grain_rolled_equiaxed'),
                                          os.path.join(DIR_LOC, r'Section_3\7500_grain\7500_grain_rolled_elongated')]  

locats_Al7075_compare_shape_41000_grain = [os.path.join(DIR_LOC, r'Section_3\41000_grain\41000_grain_cubic_equiaxed'),
                                           os.path.join(DIR_LOC, r'Section_3\41000_grain\41000_grain_cubic_elongated'),
                                           os.path.join(DIR_LOC, r'Section_3\41000_grain\41000_grain_random_equiaxed'),
                                           os.path.join(DIR_LOC, r'Section_3\41000_grain\41000_grain_random_elongated'),
                                           os.path.join(DIR_LOC, r'Section_3\41000_grain\41000_grain_rolled_equiaxed'),
                                           os.path.join(DIR_LOC, r'Section_3\41000_grain\41000_grain_rolled_elongated')]  

locats_Al7075_convergence = [os.path.dirname(DIR_LOC) + '/compile/random_periodic_equiaxed_90^3_22_total',
                             os.path.dirname(DIR_LOC) + '/compile/random_periodic_equiaxed_160^3_4_total',
                             os.path.dirname(DIR_LOC) + '/compile/random_periodic_equiaxed_200^3_2_total',
                             os.path.dirname(DIR_LOC) + '/compile/random_periodic_equiaxed_250^3_1_total']    

locats_Al7075_convergence = [os.path.join(DIR_LOC, r'Section_4\7500_x_22'),
                             os.path.join(DIR_LOC, r'Section_4\41000_x_4'),
                             os.path.join(DIR_LOC, r'Section_4\80000_x_2'),
                             os.path.join(DIR_LOC, r'Section_4\160000_x_1')]  

pad_dist = 12
font_size = 8
tick_widths = 1
tick_lens = 5


def read_FIPs_from_single_SVE(directory, num, num_fips_extract):
    # Additional unused function to read in the largest sub-band averaged FIPs (one per grain) from a single microstructure instantiation
    # This is particularly useful when considering a very large SVE with more than ~10,000 grains as this function can take a long time!
    # Go to directory
    
    tmp_dir = os.getcwd()
    os.chdir(directory)
    
    # Specify name of pickle file with sub-band averaged FIPs
    fname = 'sub_band_averaged_FS_FIP_pickle_%d.p' % num    
    
    # Specify how many of the highest FIPs per grain should be imported. Typically, only the few hundred highest FIPs are of interest
    # This significantly speeds up this algorithm!
    # IMPORTANT: If th is is set below the number of grains in the instantiation, the function will fail! 
    
    # Initialize list of just FIPs
    new_all_fs_fips = []

    # Initialize list to keep track of which grains have already been considered
    added_g = []
    
    # Read in FIPs
    h1 = open(fname,'rb')
    fips = p.load(h1, encoding = 'latin1')
    h1.close()
    
    # Sort in descending order
    sorted_fips = sorted(fips.items(), key=operator.itemgetter(1))
    sorted_fips.reverse()
    
    # Initialize array with more detailed FIP data
    # FIP, grain number, slip system number, layer number, sub-band region number
    all_data = np.zeros((num_fips_extract,5))
    
    # Main counter
    nn = 0
    
    # Track counter
    mm = 0    
    
    while len(added_g) < num_fips_extract:
    
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
        nn += 1     
    
    os.chdir(tmp_dir)
    return new_all_fs_fips
    # return new_all_fs_fips, all_data, added_g 

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
    
    # Sub-band and band averaged FIPs are extracted in the same fashion
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
   
def plot_FIPS(base_directory, plt_type, mat, num_fips_plot, FIP_type, averaging_type, num_plot_variation, save_fig = True):
    # Function to plot FIPs from bulk (i.e., fully periodic) and surface (i.e., traction-free/free surface) simulations, or other types of simulations
    print('Plotting volume averaged FIPs')
    os.chdir(base_directory)

    # Specify which FIPS to plot
    
    # Compare FIPs from microstructures with ~7,500 grains
    if mat == '7500_grain_compare_shape':
        # Names to use for figure legend
        names = ["Cubic Equiaxed", "Cubic Elongated", "Random Equiaxed", "Random Elongated", "Rolled Equiaxed", "Rolled Elongated"]
        
        # Locations of FIP files
        locs = locats_Al7075_compare_shape_7500_grain
        
        # Number of columns for figure legend
        plot_col = 1
        
        # Marker colors
        cfm = ['r','r','b','b','g','g']
        
        # Marker shapes
        sfm = ['o','o','s','s','^','^']   
        
        # If True, every other FIP dataset will be plotted with hollow markers
        hollow_and_full = True
        
        # Plot to show variability?
        variability = False
        
    # Compare FIPs from microstructures with ~41,000 grains
    elif mat == '41000_grain_compare_shape':
        # Names to use for figure legend
        names = ["Cubic Equiaxed", "Cubic Elongated", "Random Equiaxed", "Random Elongated", "Rolled Equiaxed", "Rolled Elongated"]
        
        # Locations of FIP files
        locs = locats_Al7075_compare_shape_41000_grain
        
        # Number of columns for figure legend
        plot_col = 1
        
        # Marker colors
        cfm = ['r','r','b','b','g','g']
        
        # Marker shapes
        sfm = ['o','o','s','s','^','^']   
        
        # If True, every other FIP dataset will be plotted with hollow markers
        hollow_and_full = True
        
        # Plot to show variability?
        variability = False
        
    # Plot FIPs from the four ensembles, each with a total of ~160,000 grains
    elif mat == 'Al7075_convergence':
        
        # Names to use for figure legend
        names = ['~7,500     grains x 22 SVEs', '~41,000   grains x   4 SVEs', '~80,000   grains x   2 SVEs', '~161,000 grains x   1 SVE']
        
        # Locations of FIP files
        locs = locats_Al7075_convergence
        
        # Number of columns for figure legend
        plot_col = 1
        
        # Marker colors
        cfm = ['r','b','g','m']
        
        # Marker shapes
        sfm = ['s','o','^','*']   
        
        # If True, every other FIP dataset will be plotted with hollow markers
        hollow_and_full = False   
        
        # Plot to show variability?
        variability = False
    
    # Plot the variability in the 90^3 simulations
    elif mat == '7500_variation':
    
        # Names to use for figure legend
        names = ['~7,500 grains x %d SVEs' % num_plot_variation]
        
        # Locations of FIP files
        locs = locats_Al7075_convergence[0]
        
        # Number of columns for figure legend
        plot_col = 1
        
        # Marker colors
        cfm = plt_cols.cm.Reds(np.linspace(0,1,num_plot_variation+1))[1:]
        
        # Marker shapes
        sfm = ['s'] * num_plot_variation
        
        # If True, every other FIP dataset will be plotted with hollow markers
        hollow_and_full = False       

        # Plot to show variability?
        variability = True
        
    # Plot the variability in the 160^3 simulations
    elif mat == '41000_variation':
        
        # Names to use for figure legend
        names = ['~41,000 grains x %d SVEs' % num_plot_variation]
        
        # Locations of FIP files
        locs = locats_Al7075_convergence[1]
        
        # Number of columns for figure legend
        plot_col = 1
        
        # Marker colors
        cfm = plt_cols.cm.Blues(np.linspace(0,1,num_plot_variation+1))[1:]
        
        # Marker shapes
        sfm = ['o'] * num_plot_variation
        
        # If True, every other FIP dataset will be plotted with hollow markers
        hollow_and_full = False       

        # Plot to show variability?
        variability = True    

    # Plot the variability in the 200^3 simulations
    elif mat == '80000_variation':
        
        # Names to use for figure legend
        names = ['~80,000 grains x %d SVEs' % num_plot_variation]
        
        # Locations of FIP files
        locs = locats_Al7075_convergence[2]
        
        # Number of columns for figure legend
        plot_col = 1
        
        # Marker colors
        cfm = plt_cols.cm.Greens(np.linspace(0,1,num_plot_variation+1))[1:]
        
        # Marker shapes
        sfm = ['^'] * num_plot_variation
        
        # If True, every other FIP dataset will be plotted with hollow markers
        hollow_and_full = False       

        # Plot to show variability?
        variability = True 
        
    else:
        raise ValueError('Please check "mat" variable!') 


    if variability:
        # Execute this section if interested in variability across FIP EVDs
        fip_list = []
        
        # Number of FIPs to extract from each pickle file
        num_fips_extract = 1500
        
        # Keep track of file nomenclature
        grain_per_SVE = int(mat.split('_')[0])

        # Define filename
        file_name_4 = os.path.join(base_directory, 'compiled_from_individ_SVE_FIP_data_%d.p' % grain_per_SVE)
        
        # If this file already exists, simply read it
        if os.path.exists(file_name_4):
        
            h4 = open(file_name_4, 'rb')
            fips_temp = p.load(h4)
            h4.close()
            fip_list = fips_temp
        
        # Otherwise, read in some number of highest FIPs from each file and plot each
        else:
        
            for vv in range(num_plot_variation):
                temp_fs_fips = read_FIPs_from_single_SVE(locs, vv, num_fips_extract)
                
                fip_list.append(temp_fs_fips)
                
            file_name_4 = os.path.join(base_directory, 'compiled_from_individ_SVE_FIP_data_%d.p' % grain_per_SVE)
            h4 = open(file_name_4, 'wb')
            p.dump(fip_list, h4)
            h4.close()

    else:
 
        # Number of total FIPs to extract from simulation files
        num_fips_extract = 1500
        
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
                # print(pickle_location)
                temp_fs_fips = read_pickled_SBA_FIPs(pickle_location, num_fips_extract, FIP_type, averaging_type)
                # temp_fs_fips = read_FIPs_from_single_SVE(pickle_location)
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
        fig, r_squared_values, slope_values, y_intercept_values, highest_FIP = plot_gumbels(fip_list_to_plot, plt_type, cfm, sfm, mat, plot_col, variability, names, hollow_and_full, xlabel = "ln(FS FIPs)")
    elif plt_type == 'gumbel':
        fig, r_squared_values, slope_values, y_intercept_values, highest_FIP = plot_gumbels(fip_list_to_plot, plt_type, cfm, sfm, mat, plot_col, variability, names, hollow_and_full, xlabel = "FIP")        
    
    
    if save_fig:
        if variability:
            plt.savefig("%s_EVD_%s_%s_%d_%s_num_SVEs_%d" % (FIP_type, averaging_type, plt_type, num_fips_plot, mat, num_plot_variation), dpi=fig.dpi)
        else:
            plt.savefig("%s_EVD_%s_%s_%d_%s" % (FIP_type, averaging_type, plt_type, num_fips_plot, mat), dpi=fig.dpi)
        
    # Write r-squared values to .csv file 
    fname = os.path.join(os.getcwd(),'r_squared_%s_%s_%s_%d_%s.csv' % (plt_type, FIP_type, averaging_type, num_fips_plot, mat))    
    write_r = open(fname,'w')
    for i in range(len(r_squared_values)):
        write_r.write(str(r_squared_values[i]) + ',' + str(slope_values[i]) + ',' + str(y_intercept_values[i])+ ',' + str(highest_FIP[i]) + '\n')
    write_r.close()
    print('SBA FIP r-squared value for ' + plt_type + ' is ' + str(np.average(r_squared_values)) + ' for ' + str(num_fips_plot) + ' FIPs') 
   
def prettify_frame(ax):
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
    
def plot_gumbels(data, plt_type, cfm, sfm, mat, plot_col, variability, names = None, hollow_and_full = False, xlabel = "", plot_lin = False):
    
    if names is None:
        names = [str(i) for i in range(len(data))]

    # Plot all FIPs for comparison
    
    if mat == '7500_grain_compare_shape' or mat == '41000_grain_compare_shape':
        fig = plt.figure(facecolor="white", figsize=(6, 4), dpi=1200)
    elif mat == '7500_variation' or mat == '41000_variation' or mat == '80000_variation' or mat == 'Al7075_convergence':  
        fig = plt.figure(facecolor="white", figsize=(6, 4), dpi=1200)
    else:
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
        
            if variability:
                    if i == len(data) - 1:
                        ax.scatter(d, ys, c=cfm[i], label=names[0], marker=sfm[i], alpha=0.75, linewidths=0.5, s = 50)
                    else:
                        ax.scatter(d, ys, c=cfm[i], marker=sfm[i], alpha=0.75, linewidths=0.5, s = 50)
                        
            else:
                ax.scatter(d, ys, c=cfm[i], marker=sfm[i], label=names[i], alpha=0.75, linewidths=0.5, s = 50) 
           
        # If plot_lin is true, a line of best fit is also plotted
        slope, intercept, r_value, p_value, std_err = ss.linregress(np.ndarray.tolist(d),np.ndarray.tolist(ys))
        if plt_type == 'frechet':
            interp_points = np.asarray([np.min(d), np.max(d)])
        else:
            interp_points = np.asarray([np.min(d)-np.min(d)*0.03, np.max(d)])
        # interp_points = np.asarray([np.min(d), np.max(d)])
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
        
    prettify_frame(ax)
     
    if mat == '7500_variation' or mat == '41000_variation' or mat == '80000_variation' or mat == 'Al7075_convergence':
        plt.xlim(0.44e-2, 1.25e-2)
        plt.legend(loc='lower right', fontsize='11', ncol=plot_col, framealpha=1)
        # plt.legend()
    elif mat == '41000_grain_compare_shape' or mat == '7500_grain_compare_shape':
        plt.xlim(0.32e-2, 1.0e-2)
        plt.legend(loc='lower right', fontsize='8', ncol=plot_col, framealpha=1)
    
    plt.ylabel("$-\mathrm{ln}(-\mathrm{ln}(p))$", fontsize='10')
    plt.xlabel(xlabel, fontsize='10')
    

    # Specify upper limit on plot; edit this as necessary 
    if len(d) == 100:
        plt.ylim(np.floor(min_ys), 5.5)
    elif len(d) == 150 or len(d) == 200:
        plt.ylim(np.floor(min_ys), 6)
    elif len(d) == 250:
        plt.ylim(np.floor(min_ys), 7)
    else:
        plt.ylim(np.floor(min_ys), 5)
        
    if plt_type == 'gumbel':
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText = True, useOffset=False)
    elif plt_type == 'frechet':
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText = True)
        
   
    plt.tight_layout()
    return fig, r_squared_values, slope_values, y_intercept_values, highest_FIP
    
def main(plt_type = 'gumbel'):

    # Specify folder which contains all of the simulation batch folders
    # IMPORTANT: this directory should contain the locations specified at the top of this python script, NOT the individual folders with the instantiations!
    # This script goes through EACH ONE of the folders specified and extracts the relevant information for plotting purposes
    
    # directory = os.path.dirname(DIR_LOC) + '\\PRISMS-Fatigue_tutorial'
    
    directory = os.path.join(DIR_LOC, r'plots')

    # Specify which 'material' to plot, i.e., which combinations of microsturcture folders
    # Please see the top of the "plot_FIPS" function
    mat_type = ['7500_variation', '41000_variation', '80000_variation', 'Al7075_convergence', '7500_grain_compare_shape', '41000_grain_compare_shape']
    
    for mat in mat_type:
        # Specify number of SVEs from which to plot FIPs to investigate FIP EVD variability
        if mat == '7500_variation':
            num_plot_variation = 22
            n_fip_plot = 100
        elif mat == '41000_variation':
            num_plot_variation = 4
            n_fip_plot = 100
        elif mat == '80000_variation':
            num_plot_variation = 2
            n_fip_plot = 100
        elif mat == 'Al7075_convergence':
            num_plot_variation = 1
            n_fip_plot = 100
        else:
            num_plot_variation = 1
            n_fip_plot = 50

        # Specify which FIP to import
        FIP_type = 'FS_FIP'
        
        # Specify which volume averaging scheme to import. By default, the sub-band averaged (SBA) FIPs are used
        # IMPORTANT: this will fail if the FIPs have not yet been averaged in the desired manner (see 'volume_average_FIPs.py' script)
        # Options: 'sub_band', 'band', and 'grain'
        averaging_type = 'sub_band'
        
        # Specify how many of the highest FIPs should be plotted
        # n_fip_plot = 25

        # Specify whether FIPs should be fit to the "gumbel" or "frechet" EVD
        # plt_type = 'frechet'
        
        # Call function to plot FIPs
        plot_FIPS(directory, plt_type, mat, n_fip_plot, FIP_type, averaging_type, num_plot_variation, save_fig = True)
    
if __name__ == "__main__":
    main()