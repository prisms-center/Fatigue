import os
import numpy as np
import sys
import pandas as pd
import operator
import glob
import pickle as p
import csv
import shutil

# Get name of directory that contains the PRISMS-Fatigue scripts
DIR_LOC = os.path.dirname(os.path.abspath(__file__))

def import_PRISMS_data(directory, shape, num, FIP_type, num_slip_systems, FIP_params, gamma_plane_simulations):
    # Function to import data from PRISMS-plasticity simulations and save as pickle file for volume averaging
    
    # WARNING: MANY PARTS OF THIS FUNCTION WILL CHANGE WHEN DEFINING A NEW FIP!!!
    # THIS FUNCTION CURRENTLY SUPPORTS TWO TYPES OF FIPS
    
    # Store previous directory and change directory
    prev_dir = os.getcwd()
    os.chdir(directory)
    
    # Specify files with simulation values at points of maximum compression and maximum tension
    dirr_max_comp = os.path.join(directory, 'MaxComp_%d.csv' % num)
    dirr_max_tens = os.path.join(directory, 'MaxTen_%d.csv' % num)

    # Read in data using pandas module
    aver_comp = pd.read_csv(dirr_max_comp, index_col = False, header = None)
    aver_tens = pd.read_csv(dirr_max_tens, index_col = False, header = None)
    # "index_col = False" means that the first column is NOT the index, which is the case in the quadrature output files here.
    
    
    # NOTE: the default PRISMS-Fatigue quadrature output columns correspond to the following values for each quadrature point (or element in the case of simulations with reduced integration elements)
    # The first four columns do not change during the simulation. The remaining columns correspond to current values of state variables
    # Grain ID, x position, y position, z position, plastic shear strain for slip systems 1 thru 12, stress normal to slip planes of slip systems 1 thru 12, plastic strain tensor in global directions (i.e., Ep11, Ep12, Ep13	Ep21, Ep22, Ep23, Ep31, Ep32, Ep33), Effective plastic strain


    # PRISMS can employ full integration elements, but reduced integration is employed for PRISMS-Fatigue
    # If full integration is used, please uncomment these lines of code to average values over the eight integration points per element
    # aver_comp_2 = pd.DataFrame(np.einsum('ijk->ik',aver_comp.values.reshape(-1,8,aver_comp.shape[1]))/8.0, columns=aver_comp.columns)
    # aver_tens_2 = pd.DataFrame(np.einsum('ijk->ik',aver_tens.values.reshape(-1,8,aver_tens.shape[1]))/8.0, columns=aver_tens.columns)
   
    # If reduced intergration performed in PRISMS, there is no need to average over eight integration points per element (two lines of code directly above)
    aver_comp_2 = aver_comp
    aver_tens_2 = aver_tens
    
    # Sort by X, then Y, then Z, since PRISMS discretizes the domain for parallelization.
    # Values must be sorted this way to match the way microstructures are instantiated using the generate_microstructure.py script! 
    # Positions in X, Y, and Z directions corresponds to columns 1 thru 3, respectively
    aver_comp_2_sorted = aver_comp_2.sort_values([3,2,1], ascending = [True, True, True])
    aver_tens_2_sorted = aver_tens_2.sort_values([3,2,1], ascending = [True, True, True])

    # Calculate plastic shear strain range
    # If a new FIP is specified, the column names in the .csv files will be read in according to the nomenclature below!
    # Corresponds to columns 4 thru 15
    delta_gamma     = (aver_tens_2_sorted - aver_comp_2_sorted) * 0.5
    slip_values     = abs(delta_gamma[[ii for ii in range(4,16)]])
    
    # Get stress normal to all slip planes at point of max tension
    # Corresponds to columns 16 thru 27
    normal_stresses_temp = aver_tens_2_sorted[[ii for ii in range(16,28)]]
    
    # Set any stress that is below 0 to 0 for calculation of Fatemi-Socie FIP
    normal_stresses = normal_stresses_temp.clip(lower = 0)
    
    # Delete some arrays to reduce memory consumption
    del aver_comp, aver_comp_2, aver_tens, aver_tens_2, delta_gamma, normal_stresses_temp
    
    # Initialize array of FIPs
    fips_stored = np.zeros((shape[0] * shape[1] * shape[2],num_slip_systems))
    
    # Calculate FIPs
    if FIP_type == 'FS_FIP':
        # Calculate the crystallographic version of the Fatemi-Socie FIP
        for ii in range(num_slip_systems):
            fips_stored[:,ii] = slip_values[ii + 4] * (1 + FIP_params[0] * (normal_stresses[ii + 16] / FIP_params[1]))
            
    elif FIP_type == 'plastic_shear_strain_range':
        for ii in range(num_slip_systems):
            fips_stored[:,ii] = slip_values[ii + 4]
          
    else:
        raise ValueError('Undefined FIP type specified! Please double check or formulate your FIP of choice! :)')
    
    
    # Store FIPs in pickle file
    dir_store_v1 = os.path.join(directory, 'PRISMS_%s_%d.p' % (FIP_type, num))
    h1 = open(dir_store_v1, 'wb')
    p.dump(fips_stored, h1)
    h1.close()
    
    if gamma_plane_simulations:
    
        # Calculate macroscopic plastic strain tensor, which is required to determine the response coordinate for each set of simulations on the gamma plane
        # Corresponds to columns 28 thru 36
        comp_Ep = aver_comp_2_sorted[[ii for ii in range(28,37)]]
        tens_Ep = aver_tens_2_sorted[[ii for ii in range(28,37)]]
        
        # Calculate average over instantiation
        avg_comp_Ep = comp_Ep.mean(axis=0)
        avg_tens_Ep = tens_Ep.mean(axis=0)

        # Save to .csv file
        f_write = open('Ep_averaged_%d.csv' % num, 'w', newline='')
        with f_write:
            writer = csv.writer(f_write)
            writer.writerow(['Ep11', 'Ep12', 'Ep13', 'Ep21', 'Ep22', 'Ep23', 'Ep31', 'Ep32', 'Ep33'])
            writer.writerow(avg_comp_Ep)
            writer.writerow(avg_tens_Ep)
        f_write.close()
    
    # Return to previous directory
    os.chdir(prev_dir)

def append_FIPs_to_vtk(directory, shape, num, FIP_type, num_slip_systems):
    # Function to append FIPs to .vtk file for visualization in ParaView

    # Store previous directory and change directory
    prev_dir = os.getcwd()
    os.chdir(directory)
    
    # Read in FIPs
    fname = os.path.join(directory, 'PRISMS_%s_%d.p' % (FIP_type, num))
    h1 = open(fname,'rb')
    fips = p.load(h1)
    h1.close()

    # Reshape to x, y, z, FIPs
    # Set order = 'F' because X, then Y, then Z changes
    fips_r = np.reshape(fips,(shape[0],shape[1],shape[2],num_slip_systems), order = 'F')
    
    # Find the maximum FIP per element, since there are currently as many FIPs per element as there are slip systems (because of the default FIP definition)
    fips_max_per_elem = fips_r.max(axis=3)
    fips_reshaped_back = np.reshape(fips_max_per_elem,(shape[0] * shape[1] * shape[2]), order = 'F')

    # Specify .vtk file names
    Fname_vtk_loc = os.path.join(os.getcwd(), 'Output_FakeMatl_%d.vtk' % num)
    Fname_vtk_new = os.path.join(os.getcwd(), 'Output_FakeMatl_%d_appended.vtk' % num)

    # Create copy of original .vtk file in case something goes wrong!
    shutil.copy(Fname_vtk_loc, Fname_vtk_new)

    # Open and write to .vtk
    f_vtk = open(Fname_vtk_new,'a')
    f_vtk.write('SCALARS Max_elem_FIP float 1\n')
    f_vtk.write('LOOKUP_TABLE default\n')

    # Write max element to .vtk
    counter = 0
    for kk in fips_reshaped_back:
        f_vtk.write(' %1.8f' % kk)
        counter += 1 
        if counter == 20:
            f_vtk.write('\n')
            counter = 0

    f_vtk.close()

    # Return to previous directory
    os.chdir(prev_dir)

def main():

    # Directory where PRISMS-Plasticity results .csv files are stored
    # In the case of gamma plane simulations, this should contain all the folders numbered 0 thru (number_of_folder - 1), each of which contains some number of instantiations simulated at some combination of strain state and magnitude
    directory = os.path.dirname(DIR_LOC) + '\\tutorial\\test_run_1'
    
    # directory = r'C:\Users\stopk\Documents\GitHub\PRISMS-Fatigue\tutorial\test_run_1'
    
    # Shape of microstructure instantiations (at this point, only CUBIC voxel functionality supported)
    # THIS MUST MATCH THE SHAPE FROM THE 'generate_microstructure.py' SCRIPT! 
    shape = np.asarray([29,29,29])

    # Number of microstructure instantiations for which to calculate FIPs
    # THIS MUST MATCH THE SHAPE FROM THE 'generate_microstructure.py' SCRIPT! Otherwise, FIPs will not be computed for all instantiations!
    num_instantiations = 1
   
    # Specify the FIP to be calculated; default is "FS_FIP"
    # The other FIP that can be calculated uses the plastic shear strain range on each slip system; FIP_type = 'plastic_shear_strain_range'
    FIP_type = 'FS_FIP'
    
    # Define the two Fatemi-Socie (FS) FIP parameters; Please see the references below for more information
    k_fip   = 10.0
    sigma_y = 517.0
    
    FIP_params = [k_fip, sigma_y]

    # For an fcc material, there are 12 slip systems for each element and therefore 12 FIPs per element
    num_slip_systems = 12   
    
    # Specify whether this folder contain multiple instantiations to generate the multiaxial Gamma plane
    # This requires additional post-processing of PRISMS-Plasticity output files to calculate macroscopic plastic strain tensors
    gamma_plane_simulations = False
    
    # Specify whether to append the .vtk file generated by DREAM.3D to visualize the highest FIP per element
    vtk_visualize_FIPs = False
    
    # Number of multiaxial strain state and magnitude folders, i.e., strain states simulated
    # This is only required if calculating the necessary files to generate a multiaxial gamma plane
    num_gamma_plane_folders = 1
    
    if gamma_plane_simulations:
    
        # Iterate through folders containing result files from multiaxial simulations
        for jj in range(num_gamma_plane_folders):
            dirr = os.path.join(directory, str(jj))
            print(dirr)
            
            for ii in range(num_instantiations):
                # print('Instantiation number %d' % ii)
                import_PRISMS_data(dirr, shape, ii, FIP_type, num_slip_systems, FIP_params, gamma_plane_simulations)        
        
    else:
        # Otherwise, compute FIPs for a single folder
        for ii in range(num_instantiations):
            import_PRISMS_data(directory, shape, ii, FIP_type, num_slip_systems, FIP_params, gamma_plane_simulations)      
    
    if vtk_visualize_FIPs:
        for ii in range(num_instantiations):
            # Write FIPs to .vtk file for visualization
            append_FIPs_to_vtk(directory, shape, ii, FIP_type, num_slip_systems)
    
if __name__ == "__main__":
    main()

# References with more information on these types of simulations:
# Stopka, K.S., McDowell, D.L. Microstructure-Sensitive Computational Estimates of Driving Forces for Surface Versus Subsurface Fatigue Crack Formation in Duplex Ti-6Al-4V and Al 7075-T6. JOM 72, 28–38 (2020). https://doi.org/10.1007/s11837-019-03804-1

# Stopka and McDowell, “Microstructure-Sensitive Computational Multiaxial Fatigue of Al 7075-T6 and Duplex Ti-6Al-4V,” International Journal of Fatigue, 133 (2020) 105460.  https://doi.org/10.1016/j.ijfatigue.2019.105460





