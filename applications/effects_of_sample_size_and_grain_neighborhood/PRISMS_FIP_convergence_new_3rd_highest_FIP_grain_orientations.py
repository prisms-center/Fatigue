import pandas as pd
import random
import os
import numpy as np
import xlrd
import shutil

# SCRIPT TO EDIT ORIENTATIONS OF NEAREST NEIGHBOR GRAINS CENTERED ABOUT THE GRAIN THAT MANIFESTS THE 3RD HIGHEST FIP

# This script will create new orientation_#.txt files for PRISMS-Plasticity simulations. The orientation of a single or multiple grains will be altered by drawing from other grain orientations from the 160,000 grain microstructure that is not part of the two cropped regions in Section 6 of the associated manuscript. This script will do this for the microstructure cropped about the grain that manifests the 3rd highest FIP in the 160,00 grain microstructure. 

# IMPORTANT NOTE: The microstructures simulated in the associated manuscript were generated using this script and those files are available to users. A subsequent execution of this code will create NEW AND UNIQUE microstructure files since the orientations of grain is overwritten by selecting grain orientations from the uncropped region of the 160,000 grain microstructure AT RANDOM. However, the SAME grains will be affected since they were selected based on grain size.

# Get name of directory that contains the PRISMS-Fatigue scripts, i.e., this script
DIR_LOC = os.path.dirname(os.path.abspath(__file__))

def flatten(d):
    return {i for b in [[i] if not isinstance(i, list) else flatten(i) for i in d] for i in b}

def create_new_microstructure_files_3rd_highest_FIP_grain():

    # Read in orientations from cropped 72^3 microstructure about the 3rd highest FIP grain from the 160,000 grain microstructure
    fname2 = os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\orientations_0.txt')
    df1 = pd.read_csv(fname2, delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')

    # Read in original list of grain orientations, from which we will randomly sample
    original_orientations = pd.read_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\orientations_0_original_250_250_250.txt'), delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')

    # Define grain number that manifests the 3rd highest FIP in the 160,000 grain microstructure AFTER it is cropped
    highest_FIP_grain = 555 # Indexed at 0 !

    ''' Read in list of neighbors '''
    # https://stackoverflow.com/questions/61885456/how-to-read-each-row-of-excel-file-into-a-list-so-as-to-make-whole-data-a-list-o
    
    listoflist = []
    
    # Make sure the workbook below has no header line
    workbook1 = xlrd.open_workbook(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\neighbor_list_third_grain.xlsx'))
    
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

    first_nearest_neighbors = [rr - 1 for rr in neighbor_list[highest_FIP_grain]]

    # GRAIN NUMBERS INDEXED AT 0 !!! 


    ''' Find second nearest grains '''
    second_nearest_neighbor_grains = []
    for neighbors_1 in first_nearest_neighbors:
        second_nearest_neighbor_grains.append(  [rr - 1 for rr in neighbor_list[neighbors_1]]   )
        
    flat_second = list(flatten(second_nearest_neighbor_grains))

    unique_second = []

    for item4 in flat_second:
        if item4 not in first_nearest_neighbors:
            unique_second.append(item4)

    # Remove highest FIP grain from list
    unique_second.remove(highest_FIP_grain)


    ''' Find third nearest grains '''
    third_nearest_neighbor_grains = []

    for neighbors_2 in unique_second:
        third_nearest_neighbor_grains.append(  [rr - 1 for rr in neighbor_list[neighbors_2]]   )
        
    flat_third = list(flatten(third_nearest_neighbor_grains))
        
    unique_third = []

    for item5 in flat_third:
        if (item5 not in unique_second) and (item5 not in first_nearest_neighbors):
            unique_third.append(item5)


    ''' Find fourth nearest grains '''
    fourth_nearest_neighbor_grains = []

    for neighbors_3 in unique_third:
        fourth_nearest_neighbor_grains.append(  [rr - 1 for rr in neighbor_list[neighbors_3]]   )
        
    flat_fourth = list(flatten(fourth_nearest_neighbor_grains))
        
    unique_fourth = []

    for item6 in flat_fourth:
        if (item6 not in unique_third) and (item6 not in unique_second) and (item6 not in first_nearest_neighbors):
            unique_fourth.append(item6)


    ''' Find fifth nearest grains '''
    fifth_nearest_neighbor_grains = []

    for neighbors_4 in unique_fourth:
        fifth_nearest_neighbor_grains.append(  [rr - 1 for rr in neighbor_list[neighbors_4]]   )
        
    flat_fifth = list(flatten(fifth_nearest_neighbor_grains))
        
    unique_fifth = []

    for item7 in flat_fifth:
        if (item7 not in unique_fourth) and (item7 not in unique_third) and (item7 not in unique_second):
            unique_fifth.append(item7)


    # Print number of grains in each layer

    print('Number of 1st nearest neighbors: %d' % len(first_nearest_neighbors))
    print('Number of 2nd nearest neighbors: %d' % len(unique_second))
    print('Number of 3rd nearest neighbors: %d' % len(unique_third))
    print('Number of 4th nearest neighbors: %d' % len(unique_fourth))
    print('Number of 5th nearest neighbors: %d' % len(unique_fifth))


    ''' Visualization of grain layers : '''

    Fname_vtk_loc = os.path.join(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\cropped_72_72_72_third_grain.vtk'))
    Fname_vtk_new = os.path.join(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\cropped_72_72_72_third_grain_append_grain_layers.vtk'))

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
    grain_IDs = vtk_first_part[24:-2]

    # Convert line of grain IDs from string to list of ints
    first_line = [int(s) for s in grain_IDs[0].split() if s.isdigit()]

    f_vtk = open(Fname_vtk_new,'a')

    f_vtk.write('SCALARS layers int 1\n')
    f_vtk.write('LOOKUP_TABLE default\n')

    counter = 0

    # Iterate through lines of grain IDs from .vtk file
    for kk in grain_IDs:

        # Get grain IDs
        temp_grain_IDs = [int(s) for s in kk.split() if s.isdigit()]
        
        # Iterate through each grain ID
        for jj in temp_grain_IDs:

            if (jj-1) in first_nearest_neighbors:
                f_vtk.write(' 1')
            elif (jj-1) in unique_second:
                f_vtk.write(' 2')
            elif (jj-1) in unique_third:
                f_vtk.write(' 3')
            elif (jj-1) in unique_fourth:
                f_vtk.write(' 4')
            elif (jj-1) in unique_fifth:
                f_vtk.write(' 5')
            else:
                f_vtk.write(' 0')
        
        # Write new line
        f_vtk.write('\n')

    f_vtk.close()








    # We can sample from the first ~2,000 grains since these are entirely different than the first and second nearest neighbor grains for grain 84853 of the original 250^3 microstructure! 

    df1 = pd.read_csv(fname2, delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')

    ''' Change orientations of first nearest neighbors '''
    
    num_create = 5
    for ii in range(num_create):
        
        # Create num_create of modified orientations.txt files that change the orientations of the grains that neighbor the highest FIP grain (84853, indexed at 1)
        
        # Generate list of random integers between 0 and 2000; length of first nearest neighbors
        first_while = True
        while first_while:
            randomlist_first = []
            for i in range(len(first_nearest_neighbors)):
                n = random.randint(1,1500)
                randomlist_first.append(n)

            # Check for duplicates
            first_while = any(randomlist_first.count(x) > 1 for x in randomlist_first)

        # Overwrite orientations
        for yy in range(len(first_nearest_neighbors)):
            df1.loc[first_nearest_neighbors[yy],1:3] = original_orientations.loc[randomlist_first[yy],1:3]
        
        # Write to new text file
        df1.to_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\modified_orientations_1st_NN_%d.txt' % ii), sep = ' ', index = False)



    df1 = pd.read_csv(fname2, delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')

    ''' Change orientations of second nearest neighbors '''

    num_create = 5
    for ii in range(num_create):

        # Generate list of random integers between 0 and 2000; length of first nearest neighbors
        second_while = True
        while second_while:
            randomlist_second = []
            for i in range(len(unique_second)):
                n = random.randint(1,2000)
                randomlist_second.append(n)

            # Check for duplicates
            second_while = any(randomlist_second.count(x) > 1 for x in randomlist_second)

        # Overwrite orientations
        for yy in range(len(unique_second)):
            df1.loc[unique_second[yy],1:3] = original_orientations.loc[randomlist_second[yy],1:3]
        
        # Write to new text file
        df1.to_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\modified_orientations_2nd_NN_%d.txt' % ii), sep = ' ', index = False)



    df1 = pd.read_csv(fname2, delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')

    ''' Change orientations of third nearest neighbors '''

    num_create = 5
    for ii in range(num_create):

        # Generate list of random integers between 0 and 2000; length of first nearest neighbors
        third_while = True
        while third_while:
            randomlist_third = []
            for i in range(len(unique_third)):
                n = random.randint(1,2000)
                randomlist_third.append(n)

            # Check for duplicates
            third_while = any(randomlist_third.count(x) > 1 for x in randomlist_third)

        # Overwrite orientations
        for yy in range(len(unique_third)):
            df1.loc[unique_third[yy],1:3] = original_orientations.loc[randomlist_third[yy],1:3]
        
        # Write to new text file
        df1.to_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\modified_orientations_3rd_NN_%d.txt' % ii), sep = ' ', index = False)



    df1 = pd.read_csv(fname2, delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')

    ''' Change orientations of fourth nearest neighbors '''

    num_create = 5
    for ii in range(num_create):

        # Generate list of random integers between 0 and 2000; length of first nearest neighbors
        fourth_while = True
        while fourth_while:
            # randomlist_fourth = []
            # for i in range(len(unique_fourth)):
            #     n = random.randint(1,5000)
            #     randomlist_fourth.append(n)
            
            randomlist_fourth = random.sample(range(2000),len(unique_fourth))

            # Check for duplicates
            fourth_while = any(randomlist_fourth.count(x) > 1 for x in randomlist_fourth)

        # Overwrite orientations
        for yy in range(len(unique_fourth)):
            df1.loc[unique_fourth[yy],1:3] = original_orientations.loc[randomlist_fourth[yy],1:3]
        
        # Write to new text file
        df1.to_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\modified_orientations_4th_NN_%d.txt' % ii), sep = ' ', index = False)



    df1 = pd.read_csv(fname2, delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')

    ''' Change orientations of fifth nearest neighbors '''

    num_create = 5
    for ii in range(num_create):
        
        print('On file number %d' % ii)
        
        # Generate list of random integers between 0 and 2000; length of first nearest neighbors
        fifth_while = True
        while fifth_while:
            # randomlist_fifth = []
            # for i in range(len(unique_fifth)):
                # n = random.randint(1,5000)
                # randomlist_fifth.append(n)
                
            randomlist_fifth = random.sample(range(2000),len(unique_fifth))

            # Check for duplicates
            fifth_while = any(randomlist_fifth.count(x) > 1 for x in randomlist_fifth)

        # Overwrite orientations
        for yy in range(len(unique_fifth)):
            df1.loc[unique_fifth[yy],1:3] = original_orientations.loc[randomlist_fifth[yy],1:3]
        
        # Write to new text file
        df1.to_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\modified_orientations_5th_NN_%d.txt' % ii), sep = ' ', index = False)





    ''' Extract volumes... '''
    df3 = pd.read_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\cropped_72_72_72_third_grain_first_chunk.csv'), index_col = False)
    num_elems = df3[['NumElements']].to_numpy(dtype=float)

    # Find largest grain in each layer..
    # First layer:

    largest_first_index = np.where(num_elems[first_nearest_neighbors] == max((num_elems[first_nearest_neighbors])))[0][0]
    largest_first = first_nearest_neighbors[largest_first_index]

    # Second layer:

    largest_second_index = np.where(num_elems[unique_second] == max((num_elems[unique_second])))[0][0]
    # unique_second[49]
    largest_second = unique_second[largest_second_index]
    # num_elems[largest_second]
    # This is grain number 3485, indexed at 1 !


    # Third layer:

    largest_third_index = np.where(num_elems[unique_third] == max((num_elems[unique_third])))[0][0]
    # unique_third[104]
    largest_third = unique_third[largest_third_index]
    # num_elems[largest_third]
    # This is grain number 3399, indexed at 1 !


    # Fourth layer:

    largest_fourth_index = np.where(num_elems[unique_fourth] == max((num_elems[unique_fourth])))[0][0]
    # unique_fourth[15]
    largest_fourth = unique_fourth[largest_fourth_index]
    # num_elems[largest_fourth]
    # This is grain number 4179, indexed at 1 !


    # Fifth layer:

    largest_fifth_index = np.where(num_elems[unique_fifth] == max((num_elems[unique_fifth])))[0][0]
    # unique_fifth[341]
    largest_fifth = unique_fifth[largest_fifth_index]
    # num_elems[largest_fifth]
    # This is grain number 3325, indexed at 1 !






    # Modify one grain that neighbors the grain with the largest FIP.
    # First, the grain that is largest
    df1 = pd.read_csv(fname2, delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')
    num_create = 10
    for ii in range(num_create):

        n = random.randint(1,2000)

        # Overwrite orientations

        df1.loc[first_nearest_neighbors[4],1:3] = original_orientations.loc[n,1:3]
        
        # Write to new text file
        df1.to_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\modified_orientations_largest_neighbor_%d.txt' % ii), sep = ' ', index = False)



    # Second, the grain that shares the most surface area
    df1 = pd.read_csv(fname2, delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')
    num_create = 10
    for ii in range(num_create):

        n = random.randint(1,2000)

        # Overwrite orientations

        df1.loc[3963,1:3] = original_orientations.loc[n,1:3]
        
        # Write to new text file
        df1.to_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\modified_orientations_largest_shared_area_1st_layer_%d.txt' % ii), sep = ' ', index = False)



    df1 = pd.read_csv(fname2, delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')

    ''' Modify the orientation of the largest grain in layers 2 thru 5 ! '''

    num_create = 5
    for ii in range(num_create):

        n = random.randint(1,2000)

        # Overwrite orientations

        df1.loc[largest_second,1:3] = original_orientations.loc[n,1:3]
        
        # Write to new text file
        df1.to_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\modified_orientation_largest_neighbor_in_2nd_layer_%d.txt' % ii), sep = ' ', index = False)

    df1 = pd.read_csv(fname2, delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')
    num_create = 5
    for ii in range(num_create):

        n = random.randint(1,2000)

        # Overwrite orientations

        df1.loc[largest_third,1:3] = original_orientations.loc[n,1:3]
        
        # Write to new text file
        df1.to_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\modified_orientation_largest_neighbor_in_3rd_layer_%d.txt' % ii), sep = ' ', index = False)
        
    df1 = pd.read_csv(fname2, delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')
    num_create = 5
    for ii in range(num_create):

        n = random.randint(1,2000)

        # Overwrite orientations

        df1.loc[largest_fourth,1:3] = original_orientations.loc[n,1:3]
        
        # Write to new text file
        df1.to_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\modified_orientation_largest_neighbor_in_4th_layer_%d.txt' % ii), sep = ' ', index = False)

    df1 = pd.read_csv(fname2, delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')
    num_create = 5
    for ii in range(num_create):

        n = random.randint(1,2000)

        # Overwrite orientations

        df1.loc[largest_fifth,1:3] = original_orientations.loc[n,1:3]
        
        # Write to new text file
        df1.to_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\modified_orientation_largest_neighbor_in_5th_layer_%d.txt' % ii), sep = ' ', index = False)






    ''' Mid January 2021; need to create 45 additional files that contain:
        Changed orientations for layers 3, 4, and 5; 5 sets for each one, where we change the orientation of
        1) The top 5 largest grains
        2) The 5% largest grains
        3) the 20% largest grains
        --> I.e., do not pick grains at random! '''
        
        
    # So first, let's sort the grains in each layer by volume:

    # "num_elems" contains the number of elements in each grain

    # This is how we found the largest grain in the third layer:

    
    # Number of elements in each grain of layer 3
    layer_3_num_elems_per_grain = num_elems[unique_third].astype(int).transpose()[0]
    layer_3_info = np.vstack([layer_3_num_elems_per_grain, unique_third])

    # Now sort by the number of elements in each grain!
    layer_3_sorted_info = layer_3_info.transpose()[layer_3_info.transpose()[:,0].argsort()][::-1]

    # List the grains of interest EXCLUDING the highest grain since we already changed this...!
    layer_3_largest_5_grains   = layer_3_sorted_info.transpose()[1][1:5]
    layer_3_largest_grain      = layer_3_sorted_info.transpose()[1][0]
    
    temp11 = int(np.rint(len(layer_3_sorted_info)*0.05))
    layer_3_largest_5_perc_grains   = layer_3_sorted_info.transpose()[1][1:temp11]
    
    temp22 = int(np.rint(len(layer_3_sorted_info)*0.2))
    layer_3_largest_20_perc_grains  =layer_3_sorted_info.transpose()[1][1:temp22]


    # List the grains that were changed (not including the largest) that SHOULD be changed back to what they originall were
    
    layer_3_change_back_5 = layer_3_sorted_info.transpose()[1][5:]
    
    layer_3_change_back_largest_5_perc_grains  = layer_3_sorted_info.transpose()[1][temp11:]
    layer_3_change_back_largest_20_perc_grains = layer_3_sorted_info.transpose()[1][temp22:]

   
    # Now let's modify and create new text files...!
    
    # BRILLIANT IDEA!:
    # Read in the orientations of the grains that were changed the first time around and set all other grains back to original orientations!
    
    # BETTER YET: read in the files we created last time around and CHANGE the last X amount of grains back to their original orientations! 
      

    ''' Change orientations of third nearest neighbors '''
    df1 = pd.read_csv(fname2, delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')
    num_create = 5


    for ii in range(num_create):

        # Read in previous modified orientations
        modified_orients_top_5          = pd.read_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\modified_orientations_3rd_NN_%d.txt' % ii), delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')
        modified_orients_top_5_percent  = pd.read_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\modified_orientations_3rd_NN_%d.txt' % ii), delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')
        modified_orients_top_20_percent = pd.read_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\modified_orientations_3rd_NN_%d.txt' % ii), delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')
        
        # Get orientation of largest grain changed previously...
        
        modified_orients_get_largest_ori = pd.read_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\modified_orientation_largest_neighbor_in_3rd_layer_%d.txt' % ii), delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')
        

        # Change BACK the orientations of interest!!
        modified_orients_top_5.loc[layer_3_change_back_5,1:3] = df1.loc[layer_3_change_back_5,1:3]
        modified_orients_top_5.loc[layer_3_largest_grain,1:3] = modified_orients_get_largest_ori.loc[layer_3_largest_grain,1:3]
        
        modified_orients_top_5_percent.loc[layer_3_change_back_largest_5_perc_grains,1:3] = df1.loc[layer_3_change_back_largest_5_perc_grains,1:3]
        modified_orients_top_5_percent.loc[layer_3_largest_grain,1:3] = modified_orients_get_largest_ori.loc[layer_3_largest_grain,1:3]
        
        modified_orients_top_20_percent.loc[layer_3_change_back_largest_20_perc_grains,1:3] = df1.loc[layer_3_change_back_largest_20_perc_grains,1:3]
        modified_orients_top_20_percent.loc[layer_3_largest_grain,1:3] = modified_orients_get_largest_ori.loc[layer_3_largest_grain,1:3]
        
        
        # Write to new text file
        modified_orients_top_5.to_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\3rd_highest_FIP_modify_largest_5_grains_in_third_layer_%d.txt' % ii), sep = ' ', index = False)
        
        modified_orients_top_5_percent.to_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\3rd_highest_FIP_modify_largest_5_percent_grains_in_third_layer_%d.txt'  % ii), sep = ' ', index = False)
        
        modified_orients_top_20_percent.to_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\3rd_highest_FIP_modify_largest_20_percent_grains_in_third_layer_%d.txt' % ii), sep = ' ', index = False)



    # Sweet! Now repeat for layer 4:

        
        # Number of elements in each grain of layer 4
        layer_4_num_elems_per_grain = num_elems[unique_fourth].astype(int).transpose()[0]
        layer_4_info = np.vstack([layer_4_num_elems_per_grain, unique_fourth])

        # Now sort by the number of elements in each grain!
        layer_4_sorted_info = layer_4_info.transpose()[layer_4_info.transpose()[:,0].argsort()][::-1]

        # List the grains of interest EXCLUDING the highest grain since we already changed this...!
        layer_4_largest_5_grains   = layer_4_sorted_info.transpose()[1][1:5]
        layer_4_largest_grain      = layer_4_sorted_info.transpose()[1][0]
        
        temp44 = int(np.rint(len(layer_4_sorted_info)*0.05))
        layer_4_largest_5_perc_grains   = layer_4_sorted_info.transpose()[1][1:temp44]
        
        temp55 = int(np.rint(len(layer_4_sorted_info)*0.2))
        layer_4_largest_20_perc_grains  =layer_4_sorted_info.transpose()[1][1:temp55]


        # List the grains that were changed (not including the largest) that SHOULD be changed back to what they originall were
        
        layer_4_change_back_5 = layer_4_sorted_info.transpose()[1][5:]
        
        layer_4_change_back_largest_5_perc_grains  = layer_4_sorted_info.transpose()[1][temp44:]
        layer_4_change_back_largest_20_perc_grains = layer_4_sorted_info.transpose()[1][temp55:]


    ''' Change orientations of fourth nearest neighbors '''
    df1 = pd.read_csv(fname2, delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')
    num_create = 5

    for ii in range(num_create):

        # Read in previous modified orientations
        modified_orients_top_5          = pd.read_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\modified_orientations_4th_NN_%d.txt' % ii), delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')
        modified_orients_top_5_percent  = pd.read_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\modified_orientations_4th_NN_%d.txt' % ii), delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')
        modified_orients_top_20_percent = pd.read_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\modified_orientations_4th_NN_%d.txt' % ii), delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')
        
        modified_orients_get_largest_ori = pd.read_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\modified_orientation_largest_neighbor_in_4th_layer_%d.txt' % ii), delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')

        # Change BACK the orientations of interest!!
        modified_orients_top_5.loc[layer_4_change_back_5,1:3] = df1.loc[layer_4_change_back_5,1:3]
        modified_orients_top_5.loc[layer_4_largest_grain,1:3] = modified_orients_get_largest_ori.loc[layer_4_largest_grain,1:3]
        
        modified_orients_top_5_percent.loc[layer_4_change_back_largest_5_perc_grains,1:3] = df1.loc[layer_4_change_back_largest_5_perc_grains,1:3]
        modified_orients_top_5_percent.loc[layer_4_largest_grain,1:3] = modified_orients_get_largest_ori.loc[layer_4_largest_grain,1:3]
        
        modified_orients_top_20_percent.loc[layer_4_change_back_largest_20_perc_grains,1:3] = df1.loc[layer_4_change_back_largest_20_perc_grains,1:3]
        modified_orients_top_20_percent.loc[layer_4_largest_grain,1:3] = modified_orients_get_largest_ori.loc[layer_4_largest_grain,1:3]
        
        
        # Write to new text file
        modified_orients_top_5.to_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\3rd_highest_FIP_modify_largest_5_grains_in_fourth_layer_%d.txt' % ii), sep = ' ', index = False)
        
        modified_orients_top_5_percent.to_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\3rd_highest_FIP_modify_largest_5_percent_grains_in_fourth_layer_%d.txt'  % ii), sep = ' ', index = False)
        
        modified_orients_top_20_percent.to_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\3rd_highest_FIP_modify_largest_20_percent_grains_in_fourth_layer_%d.txt' % ii), sep = ' ', index = False)



    # Sweet! Now repeat for layer 5:

        
        # Number of elements in each grain of layer 5
        layer_5_num_elems_per_grain = num_elems[unique_fifth].astype(int).transpose()[0]
        layer_5_info = np.vstack([layer_5_num_elems_per_grain, unique_fifth])

        # Now sort by the number of elements in each grain!
        layer_5_sorted_info = layer_5_info.transpose()[layer_5_info.transpose()[:,0].argsort()][::-1]

        # List the grains of interest EXCLUDING the highest grain since we already changed this...!
        layer_5_largest_5_grains   = layer_5_sorted_info.transpose()[1][1:5]
        layer_5_largest_grain      = layer_5_sorted_info.transpose()[1][0]
        
        temp77 = int(np.rint(len(layer_5_sorted_info)*0.05))
        layer_5_largest_5_perc_grains   = layer_5_sorted_info.transpose()[1][1:temp77]
        
        temp88 = int(np.rint(len(layer_5_sorted_info)*0.2))
        layer_5_largest_20_perc_grains  = layer_5_sorted_info.transpose()[1][1:temp88]


        # List the grains that were changed (not including the largest) that SHOULD be changed back to what they originall were
        
        layer_5_change_back_5 = layer_5_sorted_info.transpose()[1][5:]
        
        layer_5_change_back_largest_5_perc_grains  = layer_5_sorted_info.transpose()[1][temp77:]
        layer_5_change_back_largest_20_perc_grains = layer_5_sorted_info.transpose()[1][temp88:]


    ''' Change orientations of fourth nearest neighbors '''
    df1 = pd.read_csv(fname2, delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')
    num_create = 5

    for ii in range(num_create):

        # Read in previous modified orientations
        modified_orients_top_5          = pd.read_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\modified_orientations_5th_NN_%d.txt' % ii), delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')
        modified_orients_top_5_percent  = pd.read_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\modified_orientations_5th_NN_%d.txt' % ii), delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')
        modified_orients_top_20_percent = pd.read_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\modified_orientations_5th_NN_%d.txt' % ii), delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')
        
        modified_orients_get_largest_ori = pd.read_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\modified_orientation_largest_neighbor_in_5th_layer_%d.txt' % ii), delimiter = ' ', header = None, skiprows = 1, float_precision='round_trip')

        # Change BACK the orientations of interest!!
        modified_orients_top_5.loc[layer_5_change_back_5,1:3] = df1.loc[layer_5_change_back_5,1:3]
        modified_orients_top_5.loc[layer_5_largest_grain,1:3] = modified_orients_get_largest_ori.loc[layer_5_largest_grain,1:3]
        
        modified_orients_top_5_percent.loc[layer_5_change_back_largest_5_perc_grains,1:3] = df1.loc[layer_5_change_back_largest_5_perc_grains,1:3]
        modified_orients_top_5_percent.loc[layer_5_largest_grain,1:3] = modified_orients_get_largest_ori.loc[layer_5_largest_grain,1:3]
        
        modified_orients_top_20_percent.loc[layer_5_change_back_largest_20_perc_grains,1:3] = df1.loc[layer_5_change_back_largest_20_perc_grains,1:3]
        modified_orients_top_20_percent.loc[layer_5_largest_grain,1:3] = modified_orients_get_largest_ori.loc[layer_5_largest_grain,1:3]
        
        
        # Write to new text file
        modified_orients_top_5.to_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\3rd_highest_FIP_modify_largest_5_grains_in_fifth_layer_%d.txt' % ii), sep = ' ', index = False)
        
        modified_orients_top_5_percent.to_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\3rd_highest_FIP_modify_largest_5_percent_grains_in_fifth_layer_%d.txt'  % ii), sep = ' ', index = False)
        
        modified_orients_top_20_percent.to_csv(os.path.join(DIR_LOC, r'Section_6\NeighborHoodEffects_cropped_third_highest_FIPs_250^3\Cropped_around_3rd_highest_FIP_microstructure_data\3rd_highest_FIP_modify_largest_20_percent_grains_in_fifth_layer_%d.txt' % ii), sep = ' ', index = False)

def main():
    # Create new files for cropped microstructure about grain that manifests the 3rd highest FIP in the 160,000 grain microstructure
    create_new_microstructure_files_3rd_highest_FIP_grain()

if __name__ == "__main__":
    main()