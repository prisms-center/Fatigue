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

    print('For instantiation', num, ', the largest grain averaged FIP is:', np.max(grain_averaged_fips), 'in grain', np.argmax(grain_averaged_fips) + 1, '(indexed at 1).') 

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
            fname = os.path.join(fname1, 'sub_band_info_%d.p' % num)
            h1 = open(fname,'rb')
            master_sub_band_dictionary,number_of_layers,number_of_sub_bands = p.load(h1, encoding = 'latin1')
            h1.close()  

            # Read in which elements belong to each band
            fname = os.path.join(fname1, 'element_band_sets_%d.p' % num)
            h1 = open(fname,'rb')
            band_sets = p.load(h1, encoding = 'latin1')
            h1.close()
        else:
            fname = os.path.join(directory, 'sub_band_info_%d.p' % num)
            h1 = open(fname,'rb')
            master_sub_band_dictionary,number_of_layers,number_of_sub_bands = p.load(h1, encoding = 'latin1')
            h1.close()     
            
            # Read in which elements belong to each band
            fname = os.path.join(directory, 'element_band_sets_%d.p' % num)
            h1 = open(fname,'rb')
            band_sets = p.load(h1, encoding = 'latin1')
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

