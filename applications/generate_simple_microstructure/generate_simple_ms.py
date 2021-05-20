import os
import numpy as np

# Script to create a simple 3D microstructure of grains with some number of elements per grain, as shown in Fig. 2 of the reference below:
# G. M. Castelluccio and D. L. McDowell, "Microstructure and mesh sensitivities of mesoscale surrogate driving force measures for transgranular fatigue cracks in polycrystals," Materials Science and Engineering: A, 639, 626 (2015)
# The text file generated here is then read by the associated DREAM.3D .json pipeline to assign grain orientations.

# Users should edit the "main" function below.

# Get name of directory that contains this script
DIR_LOC = os.path.dirname(os.path.abspath(__file__))

def gen_file(directory, grains_shape, num_elem_side):
    # Function to create simple microstructure in DREAM.3D "Dx" file format
    # This can then be read in by DREAM.3D and further manipulated

    # Specify file name for microstructure
    h1 = 'grains_%d_%d_%d_elems_per_grain_%d_v1.txt' % (grains_shape[0], grains_shape[1], grains_shape[2], num_elem_side ** 3)

    # Calculate number of elements for microstructure
    ms_shape = grains_shape * num_elem_side

    # Open text file
    txt_file = open(os.path.join(directory,h1), 'w')

    # Write first portion of Dx.txt file:
    txt_file.write('# object 1 are the regular positions. The grid is %d %d %d. The origin is\n' % (ms_shape[2], ms_shape[1], ms_shape[0]))
    txt_file.write('# at [0 0 0], and the deltas are 1 in the first and third dimensions, and\n')
    txt_file.write('# 2 in the second dimension\n')
    txt_file.write('#\n')
    txt_file.write('object 1 class gridpositions counts %d %d %d\n' % (ms_shape[2], ms_shape[1], ms_shape[0]))
    txt_file.write('origin 0 0 0\n')
    txt_file.write('delta  1 0 0\n')
    txt_file.write('delta  0 1 0\n')
    txt_file.write('delta  0 0 1\n')
    txt_file.write('#\n')
    txt_file.write('# object 2 are the regular connections\n')
    txt_file.write('#\n')
    txt_file.write('object 2 class gridconnections counts %d %d %d\n' % (ms_shape[2], ms_shape[1], ms_shape[0]))
    txt_file.write('#\n')
    txt_file.write('# object 3 are the data, which are in a one-to-one correspondence with\n')
    txt_file.write('# the positions ("dep" on positions). The positions increment in the order\n')
    txt_file.write('# "last index varies fastest", i.e. (x0, y0, z0), (x0, y0, z1), (x0, y0, z2),\n')
    txt_file.write('# (x0, y1, z0), etc.\n')
    txt_file.write('#\n')
    txt_file.write('object 3 class array type int rank 0 items %d data follows\n' % (ms_shape[2] * ms_shape[1] * ms_shape[0]))

    # Calculate total number of grains
    num_grains   = np.product(grains_shape)
    
    # Create 'master' grainID matrix based on desired number of grains in the X, Y, and Z directions
    grainIDs_temp = np.arange(1,num_grains + 1)
    grainIDs      = np.reshape(grainIDs_temp, grains_shape)
    master_list   = []
    
    # Outer loop should iterate through number of elements in the X
    for xx in range(ms_shape[0]):
    
        for yy in range(ms_shape[1]):
        
            for zz in range(ms_shape[2]):

                temp_val = grainIDs[int(xx/num_elem_side),int(yy/num_elem_side),int(zz/num_elem_side)]
                
                master_list.append(temp_val)
    
    # Convert list to numpy array
    np_master_list = np.asarray(master_list)

    # Number of lines is equal to the product of voxels in the X and Y directions, i.e., X * Y, ms_shape[0] * ms_shape[1]
    # Number of values per line is equal to the number of voxels in the Z direction, ms_shape[2]

    for ii in range(ms_shape[0] * ms_shape[1]):

        grainIDs_to_write = np_master_list[ii * ms_shape[2]: (ii + 1) * ms_shape[2]]
        
        to_write = np.array2string(grainIDs_to_write, max_line_width = 9999999)[1:-1]

        txt_file.write(to_write + '\n')

    # Write bottom portion of Dx text file
    txt_file.write('attribute "dep" string "positions"\n')
    txt_file.write('#\n')
    txt_file.write('# A field is created with three components: "positions", "connections",\n')
    txt_file.write('# and "data"\n')
    txt_file.write('object "regular positions regular connections" class field\n')
    txt_file.write('component  "positions"    value 1\n')
    txt_file.write('component  "connections"  value 2\n')
    txt_file.write('component  "data"         value 3\n')
    txt_file.write('#\n')
    txt_file.write('end\n')

    # Close text file
    txt_file.close()

def main():

    # Specify directory to store files
    # Can use an absolute or local definition of the directory
    # Absolute directory path:
    # directory = r'C:\Users\stopk\Documents\GitHub\PRISMS-Fatigue\tutorial\for_simple_ms_v1'
    
    # Local directory path:
    directory = os.path.dirname(DIR_LOC) + '\\..\\tutorial\\test_ms_gen'
    
    # Create/go to desired directory
    if os.path.exists(directory):
        os.chdir(directory)
    else:
        os.makedirs(directory)
        os.chdir(directory)

    # Number of elements per side for each grain, i.e., a value of 3 will create a microstructure with 3^3 = 27 elements per grain
    num_elem_side = 6
    
    # Number of desired grains in the X, Y, and Z directions
    grains_shape = np.asarray((4,10,7))
    
    # Call function to create microstructure  
    gen_file(directory, grains_shape, num_elem_side)
    
    print('Created microstructure with %d x %d x %d grains, with %d elements per grain.' % (grains_shape[0], grains_shape[1], grains_shape[2], num_elem_side ** 3) )

if __name__ == "__main__":
    main()
   
