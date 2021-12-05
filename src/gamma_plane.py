import numpy as np
import glob as glob
import os
import matplotlib.pyplot as plt
import pickle as p
import operator
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic, ConstantKernel

DIR_LOC = os.path.dirname(os.path.abspath(__file__))

# Please read associated publication that explains generation of the gamma plane before attempting to use this script!:
# Stopka and McDowell, “Microstructure-Sensitive Computational Multiaxial Fatigue of Al 7075-T6 and Duplex Ti-6Al-4V,” 
# International Journal of Fatigue, 133 (2020) 105460
# https://doi.org/10.1016/j.ijfatigue.2019.105460

# Define extent of the gamma plane
total_extents = 0.00575

# Define colors for contour plots
colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf"]

# Define which of the strain state and magnitude ensembles (95 in total) belong to Case A and Case B.
# Note: Uniaxial strain is both Case A and B, but in this work was lumped with the Case A loading.

# Case A includes uniaxial, pure shear, and all load states inbetween as determined using the macroscopic plastic strain tensor
case_A_locs = list(range(9)) + list(range(19,50))

# Case B includes all biaxial states of strain (refer to excel spreadsheet with loading numbers and strains).
case_B_locs = list(range(9,19)) + list(range(50,80)) + list(range(81,85)) + list(range(86,95))


def read_pickled_FIPs(directory, num_fips, FIP_type, averaging_type):
    # Read in FIPs from one batch folder
    # 
    # Inputs:
    #     directory : Location of pickle files for SVE ensemble
    #     num_fips  : Number of FIPs to extract from the entire SVE ensemble
    #         (i.e., if num_fips = 50, the top 50 volume averaged FIPs from 
    #          individual grains out of the entire SVE ensemble will be reported)
    #
    # Outputs: 
    #     List of the highest 'num_fips' FIPs from the ensemble
    
    # Store current dir, change directory
    tmp_dir = os.getcwd()
    os.chdir(directory)
    
    # Find all file names with pickle FIPs
    file_names = []

    # Type of desired FIP averaging to plot
    if averaging_type == 'sub_band':
        fname = 'sub_band_averaged_%s_pickle*' % FIP_type
    elif averaging_type == 'band':
        fname = 'band_averaged_%s_pickle*' % FIP_type
    elif averaging_type == 'grain':
        fname = 'grain_averaged_%s_pickle*' % FIP_type
    else:
        raise ValueError('Unknown input! Please ensure averaging_type is "sub_band", "band", or "grain"!')

    for Name in glob.glob(fname):
        file_names.append(Name)
 
    print('Currently in %s' % directory)
    
    # Initialize list with FIPs
    new_all_fs_fips = []
    
    if averaging_type == 'sub_band' or averaging_type == 'band':
        # Iterate through files
        for kk in file_names:

            # Initialize list of grains from which the highest FIP was already extracted
            # The file from which FIPs are read is sorted in descending order
            # Thus, once a FIP is read from a grain, this grain number is stored and 
            # no other FIPs from this grain are considered.
            added_g = []
            
            # Read in FIPs
            fname1 = os.path.join(os.getcwd(),kk)
            h1 = open(fname1,'rb')
            fips = p.load(h1)
            h1.close()
            
            # Print current file name
            print(kk)
            
            # Sort FIPs
            sorted_fips = sorted(fips.items(), key=operator.itemgetter(1))
            sorted_fips.reverse()
            
            # Iterate through all FIPs, extract the maximum FIP per grain
            for nn in range(len(sorted_fips)):
            
                # If the current FIP is in a grain which is NOT yet in the list, 
                #     add this FIP to 'sorted_fips' list and add this grain number to 'added_g' list
                if sorted_fips[nn][0][0] not in added_g:
                    
                    added_g.append(sorted_fips[nn][0][0])
                    new_all_fs_fips.append(sorted_fips[nn][1])

        # Sort FIPs in descending order
        new_all_fs_fips.sort(reverse=True)
        # Change back to previous directory
        os.chdir(tmp_dir)    
        return new_all_fs_fips[0:num_fips] 
    
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

def get_gamma_FIPs(directory, num_FIPs_extract, FIP_type, averaging_type):
    # Read and pickle the highest 'num_FIPs_extract' amount of FIPs from each instantiation folder (currently set to 50)    
    #
    # Inputs:
    #     directory : Location of pickle files for SVE ensemble
    #     num_fips  : Number of FIPs to extract from the entire SVE ensemble
    #         (i.e., if num_fips = 50, the top 50 sub-band-averaged FIPs from 
    #         individual grains out of the entire SVE ensemble will be reported)
    #
    # Outputs: 
    #     None - A directory is created in which a pickle file for each simulation folder exists.
    

    # Create directory in which to save FIPs in pickle files
    save_dir = os.path.join(directory,'pickled_ensemble_fips')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Define name of pickle file with FIPs
    fips_file = 'prisms_pickled_ensemble_%s_%s_%d.p'
    
    # While iterating through all numbered folders in the main directory...
    # Variable 'i' represents number of folder
    i = 0
    
    while True:
        
        # If pickle file for a folder doesn't exist, read in FIPs and create the folder
        fname = fips_file % (FIP_type, averaging_type, i)
        
        if not os.path.exists(os.path.join(save_dir,fname)):
            
            # Check if the next simulation folder exists. If not, break While loop.
            d = os.path.join(directory, str(i))
            if not os.path.exists(d):
                break
                
            # Read FIPs from ensemble folder 'd'
            SBA_FIPs = read_pickled_FIPs(d, num_FIPs_extract, FIP_type, averaging_type)
            
            # Pickle FIPs into directory
            fname_33 = os.path.join(save_dir,fname)
            h_33 = open(fname_33,'wb')
            p.dump(SBA_FIPs,h_33)
            h_33.close()
         
        # Iterate to the next instantiation folder
        i += 1
   
def compile_statistical_info(directory, num_read, FIP_type, averaging_type):
    # Read in FIP pickle files for each simulation ensemble 
    #     and determine average values of the top 'num_read' FIPs. 
    #     Split FIPs based on whether they are Case A or Case B
    #
    # Inputs:
    #     directory : Location of pickle files for SVE ensemble
    #     num_read  : Number of FIPs to use for plotting iso-FIP contours
    #         (i.e., if num_fips = 10, the top 10 volume averaged FIPs from 
    #         an ensemble will be read and averaged to use in the Gaussian Process Regression model)
    #
    # Outputs: 
    #     a_x_y  : X and Y location for the Case A data on the Gamma Plane
    #     a_FIPs : Average value of the top 'num_read' FIPs for the Case A data
    #     b_x_y  : X and Y location for the Case B data on the Gamma Plane
    #     b_FIPs : Average value of the top 'num_read' FIPs for the Case B data    

    # Read in FIPs
    SBA_FIPs = np.asarray(read_gamma_FIPs(num_read, directory, FIP_type, averaging_type))
    
    # Read in macroscopic plastic strain for each 
    avg_x_y_form = read_avg_Ep_tensor_difference(directory)
    
    # Get x and y positions on Gamma plane
    x_y = get_xs_plastics(avg_x_y_form)
    
    # ************************** Case A Values **************************
    a_x_y  = x_y[case_A_locs]
    a_FIPs = SBA_FIPs[case_A_locs]
    
    # ************************** Case B Values **************************
    b_x_y  = x_y[case_B_locs]
    b_FIPs = SBA_FIPs[case_B_locs]
    
    return a_x_y, a_FIPs, b_x_y, b_FIPs

def get_case_a_case_b_transition(x, response):
    # Pass in [x,y] values for Gamma plane plot, fatigue lives and nuggets

    # response = np.log(response)
    
    # IMPORTANT: The variable alpha in the command below controls the amount of noise in the GPR model and may require adjustment for proper fitting of curves on the gamma plane!

    gp = GPR(kernel = 1e-7 * RationalQuadratic(), n_restarts_optimizer = 10, alpha = 5e-5)
    
    x_min = np.min(x[:,0])
 
    gp.fit(x, response)
    
    xs = np.linspace(x_min, total_extents, 16)
    ys = xs/3
    xs = np.reshape(xs, (xs.size,1))
    ys = np.reshape(ys, (ys.size,1))
    xs = np.concatenate((xs, ys), axis=1)
    top_z = gp.predict(xs)
    
    # top_z = np.exp(top_z)
    
    return xs, top_z

def plot_contours(x, response, case_b=True, ax=None):

    # Initialize GP model
    gp = GPR(kernel = 1e-7 * RationalQuadratic(), n_restarts_optimizer = 10, alpha = 5e-5)
    
    # Determine extends of Gamma plane in the X direction
    x_max = total_extents
    x_min = np.min(x[:,0])
    
    # Fit gaussian model based on observation points
    gp.fit(x, response)

    # Create 100 x 100 grid
    index = 0
    delta = .001
    x_1 = np.arange(0, 1, delta)
    y_1 = np.arange(0, 1, delta)
    X, Y = np.meshgrid(x_1, y_1)
    
    # transform square sample space into the sample space for a gamma plane (triangle)
    X = X*(x_max-x_min) + x_min
    if case_b:
        Y = (2*Y-1)*X/3
    else:
        Y = Y*X/3
    
    X_1 = np.reshape(X,(X.size,1))
    Y_1 = np.reshape(Y,(X.size,1))
    
    blarg = np.concatenate((X_1, Y_1), axis=1)
    Z_1, std = gp.predict(blarg, return_std=True)

    Z = np.reshape(Z_1, X.shape)
    std = np.reshape(std, X.shape)
    
    if ax is None:
        fig = plt.figure(facecolor="white", figsize=(22,15), dpi=1200)
    
    # Specify values to plot for the ISO-FIP contours
    levels = np.asarray([0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20])
    CS = plt.contour(X, Y, Z, levels, colors=colors, linewidths=4, dpi=1200)
    
    fmt = {}
    for l in CS.levels:
        # fmt[l] = str(((int(np.exp(l))-1)/10+1)*10)
        fmt[l] = str(l)    

    # Plot squares and circles corresponding to loading points used.
    if(case_b):
        plt.clabel(CS, CS.levels, inline=1, fmt=fmt, fontsize=20)
        plt.scatter(x[:,0],x[:,1],s=10,c='k',marker="o")
    else:
        plt.clabel(CS, [], inline=1, fmt=fmt, fontsize=20)
        valid_as = x[:,1] > 0
        plt.scatter(x[valid_as,0],x[valid_as,1],s=20,c='k',marker="s")
    
    # Plot bounds of Gamma plane
    plt.plot([0,x_max], [0,x_max/3], 'k-')
    plt.plot([0,x_max], [0,-x_max/3], 'k-')
    plt.plot([0,x_max], [0,0], 'k-')
    plt.ylim([-x_max/3,x_max/3])
    plt.xlim([0,x_max])
    plt.xlabel(r"$\frac{\epsilon^p_1-\epsilon^p_3}{2}$")
    plt.ylabel(r"$\frac{\epsilon^p_1+\epsilon^p_3}{2}$")
    ax = plt.subplot(111)
    for item in ([ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(40)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(30)
    return ax
    
def read_gamma_FIPs(num_read , directory, FIP_type, averaging_type):
    # Read in FIPs for each point on the gamma plane from pickle files
    #
    # Inputs:
    #     num_read  : Number of FIPs to use for plotting iso-FIP contours
    #         (i.e., if num_fips = 10, the top 10 sub-band-averaged FIPs from 
    #         an ensemble will be read and averaged to use in the Gaussian Process Regression model)
    #     directory : Location of pickle files for SVE ensemble
    #
    # Outputs:
    #     SBA_FIP_list: list of FIPs
    
    # Initialize list of FIPs
    SBA_FIP_list = []
    
    
    # Change to directory with FIPs stored in pickle files
    os.chdir(os.path.join(directory,'pickled_ensemble_fips'))
    
    # Define name of pickle file with FIPs
    pic_for = 'prisms_pickled_ensemble_%s_%s_%d.p'

    # While iterating through all numbered folders in the main directory...
    # Variable 'i' represents file number associated with ensemble folder
    i = 0

    while True:
    
        # Go throuh each file; break if file does not exist (end of files)
        fname = pic_for % (FIP_type, averaging_type, i)
        if not os.path.exists(fname):        
            break
            
        # Read FIPs
        h_33 = open(fname,'rb')
        temp_fips = p.load(h_33)
        h_33.close()
        
        # Append FIPs to master list of all FIPs
        
        SBA_FIP_list.append(temp_fips[0:num_read])
        
        # Iterate through all files
        i += 1
    
    # Change back to original directory
    os.chdir(directory)
    return SBA_FIP_list

def read_avg_Ep_tensor_difference(directory):
    # Read in the macroscopic plastic strain tensor for each SVE at the max compression and max tension loading cycles
    #     across which FIPs are calculated
    #
    # Inputs:
    #     directory : Location of pickle files for SVE ensemble
    #
    # Outputs:
    #     ensemble_average : Ensemble averaged (for each folder) plastic strain tensor for each ensemble folder

    # While iterating through all numbered folders in the main directory...
    # Variable 'i' represents number of folder
    i = 0
    
    # Initialize ensemble averaged plastic strain tensor
    ensemble_average = np.zeros((0,9))
    
    # Get current directory to return to
    tmp_dir = os.getcwd()
    
    while True:
    
        # Check if the next simulation folder exists. If not, break While loop.
        d = os.path.join(directory, str(i))
        print(d)
        if not os.path.exists(d):
            break
            
        # Go to directory
        os.chdir(d)      
        
        # Locate all files with plastic strain values
        # The files are named "Ep_averaged_#.csv" where # is the SVE number in the ensemble folder
        avg_Ep_files = glob.glob('*Ep_aver*')
        
        # Initialize array to calculate difference in plastic strain tensor for each individual SVE in the ensemble
        Ep_difference = np.zeros((len(avg_Ep_files),9))
        
        for rr, Ep_file in enumerate(avg_Ep_files):
            # Read in average Ep tensor for each instantiation, at the end of both loading blocks
            # print Ep_file
            
            # Open file, 'read' the first line to omit it
            f = open(Ep_file)
            f.readline()
            
            # Read in values at points of maximum compression, then maximum tension
            max_comp_plastic_strains = np.asarray([float(i) for i in f.readline().split(",")])
            max_tens_plastic_strains = np.asarray([float(i) for i in f.readline().split(",")])
            
            # Calculate the difference across the last loading cycle
            Ep_difference[rr] = max_tens_plastic_strains - max_comp_plastic_strains

        # Calculate average plastic strain tensor across all SVEs in an ensemble folder
        ensemble_average = np.concatenate((ensemble_average, np.atleast_2d(np.mean(Ep_difference, axis=0))), axis=0)    
        
        # Iterate through all folders
        i += 1    

    os.chdir(tmp_dir)
    return ensemble_average

def get_xs_plastics(plastics):
    # Determine x and y position of plastic strain for gamma plot
    # Inputs:
    #     plastics : Ensemble averaged plastic strain tensor
    # 
    # Outputs:
    #     xs : Locations on gamma plane
    
    # Change shape
    blah = np.reshape(plastics, (plastics.shape[0], 3,3))
    
    # Determine eigenvalues
    strains = np.linalg.eigvals(blah)
    strains = np.sort(strains, axis=1)
    
    # Calculate differences to get X and Y location on Gamma Plane 
    # See equation 6 in Stopka and McDowell (cited above)
    temp_1 = (strains[:,2]-strains[:,0])/2
    temp_2 =(strains[:,2]+strains[:,0])/2
    
    xs = np.concatenate((temp_1[:,np.newaxis], temp_2[:,np.newaxis]), axis=1)
    return xs
    
def plot_gamma_plane_locations(directory):
    # Plot ensemble locations for the gamma plane
    
    # Read in response coordinates based on macroscopic plastic strain tensor averaged for each folder
    avg_x_y_form = read_avg_Ep_tensor_difference(directory)
    x_y_form = get_xs_plastics(avg_x_y_form)

    # Create figure
    fig = plt.figure(facecolor="white", figsize=(5.08,3.386663), dpi=900)
    ax = fig.add_subplot(111)
    x_max = 5.75e-3
    
    # Place bounds on figure
    plt.plot([0,x_max], [0,x_max/3], 'k-', zorder=1)
    plt.plot([0,x_max], [0,-x_max/3], 'k-', zorder=1)
    plt.plot([0,x_max], [0,0], 'k-', zorder=1)
    plt.ylim([-x_max/3,x_max/3])
    plt.xlim([0,x_max+0.02*x_max])
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

    plt.xlabel(r"$\frac{\epsilon^p_1-\epsilon^p_3}{2}$", fontsize=15)
    plt.ylabel(r"$\frac{\epsilon^p_1+\epsilon^p_3}{2}$", fontsize=15)    
    
    # Uniaxial
    plt.scatter(x_y_form.transpose()[0][0:9], x_y_form.transpose()[1][0:9], marker = 'o', color = 'r', label = 'Both', s = 25, zorder=2)
    
    # Equibiaxial
    plt.scatter(x_y_form.transpose()[0][9:19], x_y_form.transpose()[1][9:19], marker = '^', color = 'b', label = 'Case B', s = 25, zorder=2)

    # Case B 
    plt.scatter(x_y_form.transpose()[0][list(range(50,80)) + list(range(81,85)) + list(range(86,95))], x_y_form.transpose()[1][list(range(50,80)) + list(range(81,85)) + list(range(86,95))], marker = '^', color = 'b', s = 25, zorder=2)      
    
    # Pure Shear
    plt.scatter(x_y_form.transpose()[0][19:29], x_y_form.transpose()[1][19:29], marker = 's', color = 'g', label = 'Case A', s = 25, zorder=2)   

    # Case A
    plt.scatter(x_y_form.transpose()[0][29:50], x_y_form.transpose()[1][29:50], marker = 's', color = 'g', s = 25, zorder=2)       
    
    # plt.tight_layout()

    plt.legend(fontsize = 'small', loc = 'upper left')
    plt.savefig(os.path.join(directory,'prisms_gamma_plane_ensemble_locations_test_demo.jpg'), dpi=fig.dpi, bbox_inches='tight')
   
def plot_gamma(directory, num_read, FIP_type, averaging_type, add_b_to_a = False, contt = True):

    # Get Case A and Case B x_y locations and FIPs  
    a_x_y, a_FIPs, b_x_y, b_FIPs = compile_statistical_info(directory, num_read, FIP_type, averaging_type)
    
    # Calculate average value of FIPs
    a_FIPs_avg = np.asarray([np.mean(a_FIPs[i]) for i in range(len(a_FIPs))])
    b_FIPs_avg = np.asarray([np.mean(b_FIPs[i]) for i in range(len(b_FIPs))])
    
    # Enforce continuity at the top edge of the Gamma plane
    if contt == True:
        if add_b_to_a:
            # Add information from Case B to Case A contours
            top_xs, top_zs = get_case_a_case_b_transition(b_x_y, b_FIPs_avg)
            a_x_y      = np.concatenate((a_x_y,      top_xs), axis=0)
            a_FIPs_avg = np.concatenate((a_FIPs_avg, top_zs), axis=0)
            descc = 'add_b_to_a'
        else:
            # Add information from Case A to Case B contours
            top_xs, top_zs = get_case_a_case_b_transition(a_x_y, a_FIPs_avg)
            b_x_y      = np.concatenate((b_x_y,      top_xs), axis=0)
            b_FIPs_avg = np.concatenate((b_FIPs_avg, top_zs), axis=0)
            descc = 'add_a_to_b'
    else:
        descc = ''
    
    ax = plot_contours(b_x_y, b_FIPs_avg)    
    plot_contours(a_x_y,a_FIPs_avg, case_b = False, ax=ax)
    plt.savefig(os.path.join(directory, "Gamma_Combined_%d_%s" % (num_read, descc)), bbox_inches='tight')  
 
def main(): 
    
    # Specify name of directory with simulation folders, each with the same X number of microstructures simulated to some combination of strain state and magnitude, and numbered 0 thru (number_of_folder - 1)
    directory = os.path.dirname(DIR_LOC) + '\\tutorial\\MultiaxialFatigue_Al7075T6'

    # Specify the number of FIPs to extract from each simulation folder
    num_FIPs_extract = 50
    
    # Specify how many of the highest FIPs from each batch of simulations should be considered to plot the ISO-FIP contours
    num_read_top_FIPs = 10

    # Specify which FIP to import
    FIP_type = 'FS_FIP'
    
    # Specify type of FIP averaging: 'sub_band', 'band', or 'grain'
    averaging_type = 'sub_band'   

    # Plot the locations of each simulation batch folder in terms of only the response coordinates (x and y position on the gamma plane)
    plot_gamma_plane_locations(directory)    
    
    # This function will extract the 50 highest fips from each folder and store these in a single pickle file.
    # This function is run once to speed up subsequent analysis and plotting
    get_gamma_FIPs(directory, num_FIPs_extract, FIP_type, averaging_type)
    
    # Plot the gamma plane
    plot_gamma(directory, num_read_top_FIPs, FIP_type, averaging_type)

if __name__ == "__main__":
    main()
    
# Relevant references:

# M. Yaghoobi, K. S. Stopka, A. Lakshmanan, V. Sundararaghavan, J. E. Allison, and D. L. McDowell. PRISMS-Fatigue computational framework for fatigue analysis in polycrystalline metals and alloys. npj Comput. Mater., 7, 38 (2021). https://doi.org/10.1038/s41524-021-00506-8

# Stopka, K.S., McDowell, D.L. Microstructure-Sensitive Computational Estimates of Driving Forces for Surface Versus Subsurface Fatigue Crack Formation in Duplex Ti-6Al-4V and Al 7075-T6. JOM 72, 28–38 (2020). https://doi.org/10.1007/s11837-019-03804-1

# Stopka and McDowell, “Microstructure-Sensitive Computational Multiaxial Fatigue of Al 7075-T6 and Duplex Ti-6Al-4V,” International Journal of Fatigue, 133 (2020) 105460.  https://doi.org/10.1016/j.ijfatigue.2019.105460

# Stopka, K.S., Gu, T., McDowell, D.L. Effects of algorithmic simulation parameters on the prediction of extreme value fatigue indicator parameters in duplex Ti-6Al-4V. International Journal of Fatigue, 141 (2020) 105865.  https://doi.org/10.1016/j.ijfatigue.2020.105865

# Castelluccio, G.M., McDowell, D.L. Assessment of small fatigue crack growth driving forces in single crystals with and without slip bands. Int J Fract 176, 49–64 (2012). https://doi.org/10.1007/s10704-012-9726-y