import os
import numpy as np
import pickle as p
import operator
import sys

# Get name of directory that contains the PRISMS-Fatigue scripts
DIR_LOC = os.path.dirname(os.path.abspath(__file__))
    
def Al7075_sub_band_averaging(element_fip, FIP_type, number_of_layers, number_of_sub_bands, master_sub_band_dictionary, num, num_grains, directory):
    # Function to average FIPs over sub-band regions
    
    # Specify number of slip planes; 4 for fcc material
    planes = 4

    # Initialize dictionary to store sub-band-averaged FIP values
    Avg_FIP = {}
    
    # Iterate over number of grains
    for ii in range(num_grains):
    
        # Iterate over planes 
        for jj in range(planes):

            # Iterate over layers 
            for kk in range(int(number_of_layers[ii,jj])):
                
                # Iterate over sub-band regions
                for mm in range(number_of_sub_bands[ii,jj,kk]):
                
                    # Read in the number and list of elements in a band
                    averaging_elements = master_sub_band_dictionary[ii,jj,kk,mm]
                    number_of_averaging_elements = len(averaging_elements)
                    
                    for slip_system_count in range(jj+(jj*2),jj+(jj*2)+3):
                        FIP_sum = 0
                        
                        # Sum FIP values for elements in the band
                        for item1 in averaging_elements:
                            FIP_temp = element_fip[item1-1,slip_system_count]
                            FIP_sum  = FIP_temp + FIP_sum

                        # Store FIP Values
                        Avg_FIP[ii,slip_system_count,kk,mm] = FIP_sum / number_of_averaging_elements

    # Store average FIPs into pickle for easy post-processing
    fname1 = os.path.join(directory, 'sub_band_averaged_%s_pickle_%d.p' % (FIP_type,num))
    h1 = open(fname1,'wb')
    p.dump(Avg_FIP, h1)
    h1.close()
    
    # Sort sub-band averaged FIPs in descending order
    sorted_fips = sorted(Avg_FIP.items(), key=operator.itemgetter(1))
    sorted_fips.reverse()


    # WARNING!: This section of code may run excessively long if very large microstructures are analyzed! If the file below is not of interest, please set the following variable to False
    
    # This is because the "if" section below writes every single sub band averaged FIP to a .csv file
    # This results in an acceptable analysis time and file size for smaller microstructure instantiations (i.e., 30 x 30 x 30 voxels and hundreds of grains)
    # However, this becomes extremely slow for very large microstructures. Therefore, it is sufficient to store the sub band averaged FIPs into the ".p" pickle format (as performed in the lines above) to be then read by other python scripts
    
    store_sub_band_info = True
    
    if store_sub_band_info:
    
        # Write the FIPs to .csv file for easy examination
        fid = os.path.join(directory, "sub_band_averaged_%s_%d.csv" % (FIP_type,num))
        f = open(fid, "w")
        f.write('SubBandAveragedFipValue' + ',' + 'Grain#' + ',' + 'SS#' + ',' + 'Layer#' + ',' +'Sub Band#' + ',' + '#ofElementsInSubBand' + '\n')    
        
        # Print sorted sub-band averaged FIPs to .csv file
        for yy in range(len(sorted_fips)):
            grain_temp    = sorted_fips[yy][0][0]
            slip_sys_temp = sorted_fips[yy][0][1]
            layer_temp    = sorted_fips[yy][0][2]
            sub_band_temp = sorted_fips[yy][0][3]
            FIP_temp      = sorted_fips[yy][1]
            
            slip_system_formation_plane = int((slip_sys_temp/3))
            num_elements_in_subband = len(master_sub_band_dictionary[grain_temp,slip_system_formation_plane,layer_temp,sub_band_temp])
            
            # Index information at 1 for csv file
            f.write(str(FIP_temp) + "," + str(grain_temp+1) + "," + str(slip_sys_temp+1) + "," + str(layer_temp+1) + "," + str(sub_band_temp+1) + "," + str(num_elements_in_subband) + "\n")
        f.close()
        
    
    # Store the single highest FIP per sub-band
    
    # WARNING!: This section of code may run excessively long if very large microstructures are analyzed! If the file below is not of interest, please set the following variable to False
    
    # This is once again because very large microstructures with many grains result in a slow analysis, because the code below has to check which FIPs from which grains have been considered for analysis
    
    store_sub_band_info_max_per_grain = True
    
    if store_sub_band_info_max_per_grain:
        fid2 = os.path.join(directory, "sub_band_averaged_highest_per_grain_%s_%d.csv" % (FIP_type,num))
        f1 = open(fid2, "w")
        f1.write('SubBandAveragedFipValue' + ',' + 'Grain#' + ',' + 'SS#' + ',' + 'Layer#' + ',' +'Sub Band#' + ',' + '#ofElementsInSubBand' + '\n')  
        
        # Print sorted sub-band averaged FIPs to .csv file
        
        # Create empty array to keep track of grain numbers
        added_g = []
        
        for yy in range(len(sorted_fips)):
            grain_temp    = sorted_fips[yy][0][0]
            slip_sys_temp = sorted_fips[yy][0][1]
            layer_temp    = sorted_fips[yy][0][2]
            sub_band_temp = sorted_fips[yy][0][3]
            FIP_temp      = sorted_fips[yy][1]
            
            if grain_temp not in added_g:
                added_g.append(grain_temp)
                
                slip_system_formation_plane = int((slip_sys_temp/3))
                num_elements_in_subband = len(master_sub_band_dictionary[grain_temp,slip_system_formation_plane,layer_temp,sub_band_temp])
                
                f1.write(str(FIP_temp) + "," + str(grain_temp+1) + "," + str(slip_sys_temp+1) + "," + str(layer_temp+1) + "," + str(sub_band_temp+1) + "," + str(num_elements_in_subband) + "\n")
                
        f1.close()        
    
    
    print('Highest sub-band averaged FIP location and value (location indexed at zero!):')
    print(sorted_fips[0])

def Al7075_band_averaging(element_fip, FIP_type, number_of_layers, band_sets, num, num_grains, directory):
    # Function to average FIPs over bands
    
    # Specify number of slip planes; 4 for fcc material
    planes = 4

    # Initialize dictionary to store band-averaged FIP values
    Avg_FIP = {}
    
    # Iterate over number of grains
    for ii in range(num_grains):
    
        # Iterate over planes
        for jj in range(planes):

            # Iterate over layers 
            for kk in range(int(number_of_layers[ii,jj])):
                
                # Read in the number and list of elements in a band
                averaging_elements = band_sets[ii,jj,kk]
                number_of_averaging_elements = len(averaging_elements)
                
                for slip_system_count in range(jj+(jj*2),jj+(jj*2)+3):
                    FIP_sum = 0
                    
                    # Sum FIP values for elements in the band
                    for item1 in averaging_elements:
                        FIP_temp = element_fip[item1-1,slip_system_count]
                        FIP_sum  = FIP_temp + FIP_sum

                    # Store FIP Values
                    Avg_FIP[ii,slip_system_count,kk] = FIP_sum / number_of_averaging_elements

    # Store average FIPs into pickle for easy post-processing
    fname1 = os.path.join(directory, 'band_averaged_%s_pickle_%d.p' % (FIP_type,num))
    h1 = open(fname1,'wb')
    p.dump(Avg_FIP, h1)
    h1.close()
    
    # Sort band averaged FIPs in descending order
    sorted_fips = sorted(Avg_FIP.items(), key=operator.itemgetter(1))
    sorted_fips.reverse()
    
    # Write the FIPs to .csv file for easy examination
    fid = os.path.join(directory, "band_averaged_%s_%d.csv" % (FIP_type,num))
    f = open(fid, "w")
    f.write('BandAveragedFipValue' + ',' + 'Grain#' + ',' + 'SS#' + ',' + 'Layer#' + ',' + '#ofElementsInBand' + '\n')    
    
    # Print sorted band averaged FIPs to .csv file
    for yy in range(len(sorted_fips)):
        grain_temp    = sorted_fips[yy][0][0]
        slip_sys_temp = sorted_fips[yy][0][1]
        layer_temp    = sorted_fips[yy][0][2]
        FIP_temp      = sorted_fips[yy][1]
        
        slip_system_formation_plane = int((slip_sys_temp/3))
        num_elements_in_band = len(band_sets[grain_temp,slip_system_formation_plane,layer_temp])
        
        # Index information at 1 for csv file
        f.write(str(FIP_temp) + "," + str(grain_temp+1) + "," + str(slip_sys_temp+1) + "," + str(layer_temp+1) + "," + str(num_elements_in_band) + "\n")
    f.close()


    # Store the single highest FIP per band
    fid2 = os.path.join(directory, "band_averaged_highest_per_grain_%s_%d.csv" % (FIP_type,num))
    f1 = open(fid2, "w")
    f1.write('BandAveragedFipValue' + ',' + 'Grain#' + ',' + 'SS#' + ',' + 'Layer#' + ',' + '#ofElementsInBand' + '\n')    
    
    # Print sorted sub-band averaged FIPs to .csv file
    
    # Create empty array to keep track of grain numbers
    added_g = []
    
    for yy in range(len(sorted_fips)):
        grain_temp    = sorted_fips[yy][0][0]
        slip_sys_temp = sorted_fips[yy][0][1]
        layer_temp    = sorted_fips[yy][0][2]
        FIP_temp      = sorted_fips[yy][1]
        
        if grain_temp not in added_g:
            added_g.append(grain_temp)
            
            slip_system_formation_plane = int((slip_sys_temp/3))
            num_elements_in_band = len(band_sets[grain_temp,slip_system_formation_plane,layer_temp])
            
            f1.write(str(FIP_temp) + "," + str(grain_temp+1) + "," + str(slip_sys_temp+1) + "," + str(layer_temp+1) + "," + str(num_elements_in_band) + "\n")
            
    f1.close() 

    print('Highest band averaged FIP location and value (location indexed at zero!):')
    print(sorted_fips[0])
    
def Al7075_grain_averaging(element_fip, FIP_type, grain_sets, num, directory):
    # Function to average FIPs over entire grains 
    
    # There are as many FIPs for each element as there are slip systems, so first, determine the maximum FIP in each element
    # This can be altered as necessary, for example if the imported FIP data already contain a single FIP per element
    max_fip_per_elem = np.max(element_fip, axis = 1)
    
    # Initialize array to store FIPs
    grain_averaged_fips = np.zeros(len(grain_sets))
    
    # Calculate the average FIP per grain
    for ii, elems in enumerate(grain_sets):
        grain_averaged_fips[ii] = np.mean(max_fip_per_elem[elems])

    # Store FIPs into pickle for easy post-processing
    fname1 = os.path.join(directory, 'grain_averaged_%s_pickle_%d.p' % (FIP_type,num))
    h1 = open(fname1,'wb')
    p.dump(grain_averaged_fips, h1)
    h1.close()
    
    # Write the FIPs to .csv file for easy examination
    np.savetxt(os.path.join(directory, 'grain_averaged_%s_%d.csv' % (FIP_type,num)), grain_averaged_fips, delimiter=',')

    print('For instantiation', num,', the largest grain averaged FIP is:', np.max(grain_averaged_fips), 'in grain', np.argmax(grain_averaged_fips) + 1, '(indexed at 1).') 

def call_averaging(num, directory, FIP_type, averaging_type, gamma_plane_simulations):

    # Read in FIPs for each element
    dir_store = os.path.join(directory, 'PRISMS_%s_%d.p' % (FIP_type, num))
    h1 = open(dir_store, 'rb')
    element_fip = p.load(h1)
    h1.close()

    if averaging_type == 'sub_band':
        # Average over sub bands
        
        # Read in sub-band averaging information
        
        if gamma_plane_simulations:
        
            # If volume averaging the same instantiations for gamma plane, then read info from different folder
            fname1 = os.path.join(os.path.split(directory)[0], 'instantiation_data')
            fname = os.path.join(fname1, 'sub_band_info_%d.p' % num)
            h1 = open(fname,'rb')
            master_sub_band_dictionary,number_of_layers,number_of_sub_bands = p.load(h1, encoding = 'latin1')
            h1.close()        
        else:
            fname = os.path.join(directory, 'sub_band_info_%d.p' % num)
            h1 = open(fname,'rb')
            master_sub_band_dictionary,number_of_layers,number_of_sub_bands = p.load(h1, encoding = 'latin1')
            h1.close()

        # Determine number of grains
        num_grains = len(number_of_layers)    
    
        # Call function to average FIPs over sub-band regions
        Al7075_sub_band_averaging(element_fip, FIP_type, number_of_layers, number_of_sub_bands, master_sub_band_dictionary, num, num_grains, directory)

    elif averaging_type == 'band':
        # Average over bands
        
        # Read in band averaging information; need number of layers
        
        if gamma_plane_simulations:
            fname1 = os.path.join(os.path.split(directory)[0], 'instantiation_data')
            fname = os.path.join(fname1, 'element_band_sets_%d.p' % num)
            h1 = open(fname,'rb')
            band_sets, number_of_layers = p.load(h1)
            h1.close()
        else:
            fname = os.path.join(directory, 'element_band_sets_%d.p' % num)
            h1 = open(fname,'rb')
            band_sets, number_of_layers = p.load(h1)
            h1.close()        
            
        # Determine number of grains
        num_grains = len(number_of_layers)    
        
        # Call function to average FIPs over bands
        Al7075_band_averaging(element_fip, FIP_type, number_of_layers, band_sets, num, num_grains, directory)
        
    elif averaging_type == 'grain':
        # Average over grains
        
        # Read in which elements belong to each grain
        
        if gamma_plane_simulations:
            fname1 = os.path.join(os.path.split(directory)[0], 'instantiation_data')
            fname = os.path.join(fname1, 'element_grain_sets_%d.p' % num)
            h1 = open(fname,'rb')
            grain_sets = p.load(h1, encoding = 'latin1')
            h1.close()            
        else:
            fname = os.path.join(directory, 'element_grain_sets_%d.p' % num)
            h1 = open(fname,'rb')
            grain_sets = p.load(h1, encoding = 'latin1')
            h1.close()    
        
        # Call function to average FIPs over entire grains
        Al7075_grain_averaging(element_fip, FIP_type, grain_sets, num, directory)
    else:
        raise ValueError('Please enter one of the three types of FIP averaging options!')

def read_FIPs_from_single_SVE(directory):
    # Additional unused function to read in the largest sub-band averaged FIPs (one per grain) from a single microstructure instantiation
    # This is particularly useful when considering a very large SVE with more than ~10,000 grains as this function can take a long time!
    # Therefore, only a limited number of FIPs are extracted; see variable get_num_FIPs
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
    
    os.chdir(tmp_dir)

    # Specify what should be exported
    # This function is most useful in python interactive mode
    # return new_all_fs_fips, all_data, added_g 
    return new_all_fs_fips
    
def main():

    # Directory where microstructure FIP .p files are stored:
    directory = os.path.dirname(DIR_LOC) + '\\tutorial\\MultiaxialFatigue_Al7075T6'

    # directory = r'C:\Users\stopk\Documents\GitHub\PRISMS-Fatigue\tutorial\test_run_1'
    
    # Number of microstructure instantiations in each folder for which to volume average FIPs
    num_instantiations = 10
    
    # Specify the FIP to be volume averaged
    FIP_type = 'FS_FIP'
    
    # Specify type of FIP averaging: 'sub_band', 'band', or 'grain'
    averaging_type = 'sub_band'
    
    # Specify whether this folder contain multiple instantiations to generate the multiaxial Gamma plane
    gamma_plane_simulations = False
    
    # Number of multiaxial strain state and magnitude folders, i.e., strain states simulated
    # This is only required if calculating necessary files to generate multiaxial gamma plane
    num_gamma_plane_folders = 1
    
    if gamma_plane_simulations:
        # Iterate through folders containing result files from multiaxial simulations
        for jj in range(num_gamma_plane_folders):
            dirr = os.path.join(directory, str(jj))
            print(dirr)
            
            for ii in range(num_instantiations):
                call_averaging(ii, dirr, FIP_type, averaging_type, gamma_plane_simulations)     
       
    else:
        # Call function to perform averaging for each instantiation in folder
        for ii in range(num_instantiations):
            call_averaging(ii, directory, FIP_type, averaging_type, gamma_plane_simulations)

if __name__ == "__main__":
    main()

