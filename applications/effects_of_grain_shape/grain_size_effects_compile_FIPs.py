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

# Users can simply execute this script as "python compile_FIPs_grain_size_paper.py" in a command prompt window or by using an interactive python window.
# This script should be placed in the directory that contains the "NoGrainSize", "WithGrainSize," etc. directories that were downloaded from the Materials Commons


# Get name of directory that contains this script
DIR_LOC = os.path.dirname(os.path.abspath(__file__))

# Define location where FIP data is stored
GS_paper_cubic = [os.path.join(DIR_LOC, r'NoGrainSize\Cube_texture\no_gs_cubic_AR_1'),
                  os.path.join(DIR_LOC, r'NoGrainSize\Cube_texture\no_gs_cubic_AR_6'),
                  os.path.join(DIR_LOC, r'NoGrainSize\Cube_texture\no_gs_cubic_AR_6\no_gs_cubic_inst_1'),
                  os.path.join(DIR_LOC, r'WithGrainSize\Cube_texture\gs_cubic_AR_1'),
                  os.path.join(DIR_LOC, r'WithGrainSize\Cube_texture\gs_cubic_AR_6'),
                  os.path.join(DIR_LOC, r'WithGrainSize\Cube_texture\gs_cubic_AR_6\gs_cubic_inst_1')]
                  
GS_paper_random = [os.path.join(DIR_LOC, r'NoGrainSize\Random_texture\no_gs_random_AR_1'),
                   os.path.join(DIR_LOC, r'NoGrainSize\Random_texture\no_gs_random_AR_6'),
                   os.path.join(DIR_LOC, r'NoGrainSize\Random_texture\no_gs_random_AR_6\no_gs_random_inst_1'),
                   os.path.join(DIR_LOC, r'WithGrainSize\Random_texture\gs_random_AR_1'),
                   os.path.join(DIR_LOC, r'WithGrainSize\Random_texture\gs_random_AR_6'),
                   os.path.join(DIR_LOC, r'WithGrainSize\Random_texture\gs_random_AR_6\gs_random_inst_1')]
                   
GS_paper_rolled_X = [os.path.join(DIR_LOC, r'NoGrainSize\Rolling_texture\no_gs_rolled_AR_1'),
                     os.path.join(DIR_LOC, r'NoGrainSize\Rolling_texture\no_gs_rolled_AR_6'),
                     os.path.join(DIR_LOC, r'NoGrainSize\Rolling_texture\no_gs_rolled_AR_6c'),
                     os.path.join(DIR_LOC, r'WithGrainSize\Rolling_texture\gs_rolled_AR_1'),
                     os.path.join(DIR_LOC, r'WithGrainSize\Rolling_texture\gs_rolled_AR_6'),
                     os.path.join(DIR_LOC, r'WithGrainSize\Rolling_texture\gs_rolled_AR_6c')]
                     
GS_paper_rolled_Z = [os.path.join(DIR_LOC, r'NoGrainSize\Rolling_texture\no_gs_rolled_AR_1\no_gs_rolled_z_AR_1'),
                     os.path.join(DIR_LOC, r'NoGrainSize\Rolling_texture\no_gs_rolled_AR_6\no_gs_rolled_z_AR_6'),
                     os.path.join(DIR_LOC, r'NoGrainSize\Rolling_texture\no_gs_rolled_AR_6c\no_gs_rolled_z_AR_6c'),
                     os.path.join(DIR_LOC, r'WithGrainSize\Rolling_texture\gs_rolled_AR_1\gs_rolled_z_AR_1'),
                     os.path.join(DIR_LOC, r'WithGrainSize\Rolling_texture\gs_rolled_AR_6\gs_rolled_z_AR_6'),
                     os.path.join(DIR_LOC, r'WithGrainSize\Rolling_texture\gs_rolled_AR_6c\gs_rolled_z_AR_6c')]

pad_dist = 12
font_size = 8
tick_widths = 1
tick_lens = 5

def read_d3d_csv(directory, num):
    """
    Read grain information from a DREAM.3D formatted csv file.

    Currently uses Phases, EquivalentDiameters, EulerAngles_0, EulerAngles_1, EulerAngles_2 as headings of interest.
    This function can be easily modified to read in additional data.

    Parameters
    ----------
    filename : str or unicode
        Path to file of interest

    Returns
    -------
    phases : ndarray
        List of phases for each grain
    diameters : ndarray
        List of equivalent diameters for each grain
    orientations : ndarray
        nx3 array of Bunge Euler angles for each grain

    """
    filename =  os.path.join(directory, 'FeatureData_FakeMatl_%d.csv' % num)
    
    stat_names = ['Phases', 'EquivalentDiameters', 'EulerAngles_0',
                  'EulerAngles_1', 'EulerAngles_2']
    f = open(filename, 'r')
    num_grains = int(f.readline().split(',')[0])
    header = f.readline()
    data = header.split(',')
    remap = [0]*len(stat_names)
    for i, key in enumerate(stat_names):
        for j, header_name in enumerate(data):
            if header_name.find(key)!= -1:
                remap[i] = j
    statistics = np.zeros((0, len(remap)))
    for i in range(int(num_grains)):
        line = f.readline()
        temp = np.asarray(line.split(","))
        temp = temp[remap]
        temp = np.reshape(temp, (1, len(temp)))
        statistics = np.concatenate((statistics, temp), axis=0)
    f.close()
    statistics = statistics.astype(float)
    phases = statistics[:, 0] - 1
    phases = phases.astype(int)
    diameters = statistics[:, 1]
    orientations = statistics[:, 2:5]
    return phases, diameters, orientations

def read_FIPs_from_single_SVE(directory):
    # Additional unused function to read in the largest sub-band averaged FIPs (one per grain) from a single microstructure instantiation
    # This is particularly useful when considering a very large SVE with more than ~10,000 grains as this function can take a long time!
    # Go to directory
    
    tmp_dir = os.getcwd()
    os.chdir(directory)
    
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
    fips = p.load(h1, encoding = 'latin1')
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
            # print(mm)
        nn += 1     
    
    # Store more detailed information of grains'
    fname = 'detailed_fip_info.p'
    h1 = open(fname, 'wb')
    p.dump([new_all_fs_fips, all_data, added_g], h1)
    h1.close()
    
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
   
def plot_FIPS(base_directory, plt_type, mat, num_fips_plot, FIP_type, averaging_type, save_fig = True):
    # Function to plot FIPs from bulk (i.e., fully periodic) and surface (i.e., traction-free/free surface) simulations, or other types of simulations
    print('Plotting volume averaged FIPs')
    os.chdir(base_directory)

    # Specify which FIPS to plot
    if mat == 'gs_paper_cubic':
    
        names = ['AR 1', 'AR 6', r'AR $\frac{1}{6}$', 'AR 1 - GS', 'AR 6 - GS', r'AR $\frac{1}{6}$ - GS']
        linestyless = ['-', '-', '-', ':', ':', ':']

        locs = GS_paper_cubic
        
        plot_col = 2

        sfm = ['o','*','*','o','*','*']   
        cfm = ['r', 'b', 'g', 'r', 'b', 'g']
        
        hollow_and_full = True   

    elif mat == 'gs_paper_random':

        names = ['AR 1', 'AR 6', r'AR $\frac{1}{6}$', 'AR 1 - GS', 'AR 6 - GS', r'AR $\frac{1}{6}$ - GS']
        linestyless = ['-', '-', '-', ':', ':', ':']
        
        locs = GS_paper_random
        
        plot_col = 2
        
        sfm = ['s','X','X','s','X','X']   
        cfm = ['r', 'b', 'g', 'r', 'b', 'g']
        
        hollow_and_full = True   
 
    elif mat == 'gs_paper_rolled_x':

        names = ['AR 1', 'AR 6', r'AR $\frac{1}{6}$', 'AR 1 - GS', 'AR 6 - GS', r'AR $\frac{1}{6}$ - GS']
        linestyless = ['-', '-', '-', ':', ':', ':']

        locs = GS_paper_rolled_X
        
        plot_col = 2
        
        sfm = ['^','p','D','^','p','D']
        cfm = ['r', 'b', 'g', 'r', 'b', 'g']
        
        hollow_and_full = True   

    elif mat == 'gs_paper_rolled_z':

        names = ['AR 1', 'AR 6', r'AR $\frac{1}{6}$', 'AR 1 - GS', 'AR 6 - GS', r'AR $\frac{1}{6}$ - GS']
        linestyless = ['-', '-', '-', ':', ':', ':']
        

        locs = GS_paper_rolled_Z
        
        plot_col = 2
        
        sfm = ['^','p','D','^','p','D']
        cfm = ['r', 'b', 'g', 'r', 'b', 'g']
        
        hollow_and_full = True   

    else:
        raise ValueError('Please check "mat" variable!') 
        
        

    # Number of total FIPs to extract from simulation files
    num_fips_extract = 2500
    
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
            print(pickle_location)
            # temp_fs_fips = read_pickled_SBA_FIPs(pickle_location, num_fips_extract, FIP_type, averaging_type)
            temp_fs_fips = read_FIPs_from_single_SVE(pickle_location)
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
    
    if names is None:
        names = [str(i) for i in range(len(data))]

    # Plot all FIPs for comparison
    
    if mat == 'gs_paper_cubic' or mat == 'gs_paper_random' or mat == 'gs_paper_rolled_x' or mat == 'gs_paper_rolled_z':
        fig = plt.figure(facecolor="white", figsize=(6, 4), dpi=1200)
    else:
        fig = plt.figure(facecolor="white", figsize=(5, 3.333), dpi=1200)
        
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
            
            if mat == 'gs_paper_cubic' or mat == 'gs_paper_random' or mat == 'gs_paper_rolled_x' or mat == 'gs_paper_rolled_z':
        
                if i < 3:
                    ax.scatter(d, ys, c=cfm[i], label=names[i], marker=sfm[i], alpha=0.75, linewidths=0.5)
                
                else:
                    ax.scatter(d, ys, c='white', edgecolors=cfm[i], label=names[i], marker=sfm[i], alpha=0.75, linewidths=1)   
                    
            
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
        

    if mat == 'gs_paper_cubic' or mat == 'gs_paper_random' or mat == 'gs_paper_rolled_x' or mat == 'gs_paper_rolled_z':
        plt.xlim(0.37e-2, 1.2e-2)
    
    plt.ylabel("$-\mathrm{ln}(-\mathrm{ln}(p))$", fontsize='8')
    plt.xlabel(xlabel, fontsize='8')
    

    # Specify upper limit on plot; edit this as necessary 
    if len(d) == 100:
        plt.ylim(np.floor(min_ys), 5.5)
    elif len(d) == 150 or len(d) == 200:
        plt.ylim(np.floor(min_ys), 6)
    elif len(d) == 250:
        plt.ylim(np.floor(min_ys), 7)
    else:
        t = 4
        plt.ylim(np.floor(min_ys), 5)
        
    if plt_type == 'gumbel':
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText = True, useOffset=False)
    elif plt_type == 'frechet':
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText = True)
        
    prettify_frame(ax)
    if mat == 'gs_paper_cubic' or mat == 'gs_paper_random' or mat == 'gs_paper_rolled_x' or mat == 'gs_paper_rolled_z':
        plt.legend(fontsize='10', ncol=plot_col, framealpha=1)
    else:
        plt.legend(loc='lower right', fontsize='7', ncol=plot_col, framealpha=1)
    plt.tight_layout()
    return fig, r_squared_values, slope_values, y_intercept_values, highest_FIP

def plot_fips_vs_d():
    # Plot FIPs vs grain size
    # Mainly of interest for the random texture and AR 1
    random_no_gs = os.path.join(DIR_LOC, r'NoGrainSize\Random_texture\no_gs_random_AR_1')
    random_gs    = os.path.join(DIR_LOC, r'WithGrainSize\Random_texture\gs_random_AR_1')
    
    # Read in data for random texture with no grain size effects
    fname_no_gs = os.path.join(DIR_LOC, r'NoGrainSize\Random_texture\no_gs_random_AR_1\detailed_fip_info.p')
    h1 = open(fname_no_gs, 'rb')
    no_gs_data = p.load(h1)
    h1.close()
    
    # Read in data for random texture with grain size effects
    fname_no_gs = os.path.join(DIR_LOC, r'WithGrainSize\Random_texture\gs_random_AR_1\detailed_fip_info.p')
    h1 = open(fname_no_gs, 'rb')
    gs_data = p.load(h1)
    h1.close()
    
    # Read in grain size data for the random textured microstructure with AR 1
    inst_data_loc = os.path.join(DIR_LOC, r'plasticity_ellipsoid_Microstructures\Random_texture\Different_aspect_ratios\AR_1')
    inst_data = pd.read_csv(inst_data_loc)
    
    # GRAIN DATA AS READ IN FROM .P FILES ARE INDEXED AT 0!
    phases, diameters, orientations = read_d3d_csv(inst_data_loc, 0)
    
    gs_grain_numbers    = [x + 1 for x in gs_data[2]]
    no_gs_grain_numbers = [x + 1 for x in no_gs_data[2]]
    
    gs_data_grain_diameters    = diameters[gs_grain_numbers]
    no_gs_data_grain_diameters = diameters[no_gs_grain_numbers]
    
    
    # Plot this number of top FIPs
    top_num_FIPs = 50
    
    
    # Plot FIPs vs grain diameter
    fig = plt.figure(facecolor="white", figsize=(6, 4), dpi=1200)
    
    plt.scatter(gs_data_grain_diameters[0:top_num_FIPs],    gs_data[0][0:top_num_FIPs],    color = 'b', zorder = 2, s = np.sqrt(np.arange(1,top_num_FIPs)), label = 'GS')
    plt.scatter(no_gs_data_grain_diameters[0:top_num_FIPs], no_gs_data[0][0:top_num_FIPs], color = 'r', zorder = 2, s = np.sqrt(np.arange(1,top_num_FIPs)), label = 'No GS', marker = '^')

    plt.legend(framealpha = 1)
    
    # plt.show()
    plt.ylabel('FIP')   
    plt.grid(True, zorder = 1)
    plt.xlabel('Grain diameter [microns]')
    plt.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0), useMathText = True)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_LOC, r'plots/fips_vs_grain_size_gs_model_%d' % top_num_FIPs))
    plt.close()    
    
    

    # Plot histograms of grain size
    n_bins = 50
    
    fig = plt.figure(facecolor="white", figsize=(6, 4), dpi=1200)
    
    n,x1,_ = plt.hist(diameters, bins = n_bins, histtype = 'step', color = 'white')
    n1 = n.astype(int)
    bin_centers = 0.5*(x1[1:]+x1[:-1])
    plt.plot(bin_centers, n1, color = 'b', linestyle = '-')    

    # plt.show()
    # plt.ylabel('Num')   
    plt.grid(True)
    # plt.xlabel('Grain size [microns]')     
    # plt.ticklabel_format(style = 'sci', axis='x', scilimits=(0,0), useMathText = True)    
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_LOC, r'plots/grain_diameters'))
    plt.close()     
   
def main(n_fip_plot = 100, plt_type = 'gumbel'):
    
    # Specify folder which contains all of the simulation batch folders
    # This script goes through EACH ONE of the folders specified and extracts the relevant information for plotting purposes
    
    directory = os.path.join(DIR_LOC, r'plots')

    if os.path.exists(directory):
        os.chdir(directory)
    else:
        os.makedirs(directory)
        os.chdir(directory)
        
    # Specify which 'material' to plot, i.e., which combinations of microsturcture folders
    # Please see the top of the "plot_FIPS" function
    
    mat = ['gs_paper_cubic', 'gs_paper_random', 'gs_paper_rolled_x', 'gs_paper_rolled_z']

    # Specify which FIP to import
    FIP_type = 'FS_FIP'
    
    # Specify which volume averaging scheme to import. By default, the sub-band averaged (SBA) FIPs are used
    # IMPORTANT: this will fail if the FIPs have not yet been averaged in the desired manner (see 'volume_average_FIPs.py' script)
    # Options: 'sub_band', 'band', and 'grain'
    averaging_type = 'sub_band'
    
    # Iterate through the four loading conditions specified above
    for mat_type in mat:
        plot_FIPS(directory, plt_type, mat_type, n_fip_plot, FIP_type, averaging_type, save_fig = True)
    

if __name__ == "__main__":
    main()