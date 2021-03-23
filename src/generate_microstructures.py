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

# Get name of directory that contains the PRISMS-Fatigue scripts
DIR_LOC = os.path.dirname(os.path.abspath(__file__))

# Templates for file names
PRISMS_GRAIN_ID           = "grainID_%d.txt"
PRISMS_ORIENTATION_ID     = "orientations_%d.txt"
BAND_PLANE_NORMALS        = 'band_plane_normals_%d.txt'
BAND_SETS_FORMAT          = 'band_sets_%d.txt'
BAND_SET_STATISTICS       = 'elements_per_band_%d.txt'
CSV_DREAM3D               = "FeatureData_FakeMatl_%d.csv"
EL_CENTROID_PICKLE_FORMAT = 'El_pos_%d.p' 

# Define octahedral slip system plane normal directions for fcc material
FCC_OCT_NORMALS = np.asarray([[1,1,1],
[-1,1,1],
[1,1,-1],
[1,-1,1]]) / (3**0.5)


def write_el_centroids_pickled(el_centroids, directory, num):
    # Store the centroid of each element in pickle format
    out_f = os.path.join(directory, EL_CENTROID_PICKLE_FORMAT % num)
    h1 = open(out_f, 'wb')
    p.dump(el_centroids,h1)
    h1.close()

def batch_dream3d(runner_path, pipeline_path, d3d_input_file, directory, shape, size, num_runs, is_periodic=True):
    """
    Runs DREAM.3D to generate multiple microstructure instantiations of the same basic statistics.

    Execute a set DREAM.3D pipeline over multiple instances. Create numbered output files to be read and used later.
    VTK and CSV data formats used as intermediate file outputs. CSV file format contains grain information (orientation, phase, etc.) while VTK relates voxel locations in space to the associated grain number.
    Supports commonly used variation of number of elements, element size, as well as allowing periodicity or not.
    Supports older v4 and v5 pipeline formats as well as the newer v6 pipeline JSON format.
    Created microstructures will be numbered 0-num_runs-1
    Microstructures are allowed a limited number of attempts to create, after which an IOError is raised to indicate a problem in the pipeline.

    Parameters
    ----------
    runner_path : str or unicode
        path to the pipeline_runner.exe from DREAM.3D distribution
    pipeline_path : str or unicode
        path to the pipeline file to use for this synthetic reconstruction
    d3d_input_file : str or unicode
        the .dream3d file used as input statistics for the synthetic microstructure
    directory : str or unicode
        path where intermediate files will be created (.csv and .vtk and others)
    shape : list
        List of ints for the number of voxels along each dimension of the reconstructions. Should be 3D as DREAM.3D will create a 3D microstructure.
    size : list
        List of floats for the geometric extents of the microstructure representation along each dimension. Should be in mm.
    num_runs : int
        number of synthetic reconstructions to create with these input parameters
    is_periodic : bool
        Create a periodic microstructure along all dimensions, or not.
    Returns
    -------
    csv_list : list
        List of csv file names for the created microstructure instantations.
    vtk_list : list
        List of vtk file names for the created microstructure instantations.

    """
    

    # Sub slashes in statsfile path
    d3d_input_file = d3d_input_file.replace("\\", "\\\\")

    # Set up parameters to instantiate microstructures
    res = np.asarray(size) / np.asarray(shape) * 1000
    x_elem = shape[0]
    y_elem = shape[1]
    z_elem = shape[2]

    
    if not os.path.exists(directory):
        os.makedirs(directory)
    pipeline_f = os.path.basename(pipeline_path)
    v_6 = False
    if (pipeline_f.find(".json") != -1):
        v_6 = True
    new_pipeline = os.path.join(directory, pipeline_f)
    if os.path.exists(new_pipeline):
        os.remove(new_pipeline)
    shutil.copyfile(pipeline_path, new_pipeline)
    

    
    csv = "GoalAttributes_FakeMatl_%d.csv"
    ftd = "FeatureData_FakeMatl_%d.csv"
    d3d = "Output_FakeMatl_%d.dream3d"
    inp = "ABAQUS_%d.inp"
    vtk = "Output_FakeMatl_%d.vtk"
    grain_ID_format     = "grainID_%d.txt"
    ext_grain_ID        = "_%d.txt"
    orientations_format = "orientations_%d.txt"


    vtk_list = []
    csv_list = []

    for i in range(num_runs):
        # Edit output file names so that we don't overwrite information
        if (v_6):
            directory = directory.replace("\\", "/")
            new_csv = directory + "/" + csv % i
            new_ftd = directory + "/" + ftd % i
            new_d3d = directory + "/" + d3d % i
            new_inp = directory + "/" + inp % i
            new_vtk = directory + "/" + vtk % i
            new_grain_ID_PRISMS     = directory + "/" + grain_ID_format % i
            new_grain_ID_PRISMS_ext = ext_grain_ID % i
            new_orientations_PRISMS = directory + "/" + orientations_format % i
            res_flag = False
            dim_flag = False
        else:
            new_csv = directory + "\\" + csv % i
            new_ftd = directory + "\\" + ftd % i
            new_d3d = directory + "\\" + d3d % i
            new_inp = directory + "\\" + inp % i
            new_vtk = directory + "\\" + vtk % i
        print("Current output microstructure file: " + new_vtk)
        vtk_list.append(vtk % i)
        csv_list.append(ftd % i)
        
        replace_flag_1 = True
        replace_flag_2 = True
        for line in fileinput.input(new_pipeline, inplace=True):
            
            # Go through the DREAM.3D .json file and edit file paths accordingly
            if (v_6):
                if (line.find("\"FeatureDataFile\"") != -1):
                    print("        \"FeatureDataFile\": \"%s\"," % new_ftd)
                elif (line.find("\"FileExtension\"") != -1):
                    print("        \"FileExtension\": \"%s\"," % new_grain_ID_PRISMS_ext)
                elif (line.find("\"OutputPath\"") != -1):
                    print("        \"OutputPath\": \"%s\"," % directory)

                elif (line.find("\"OutputFile\"") != -1):
                    if replace_flag_1:   
                        print("        \"OutputFile\": \"%s\"," % new_vtk)
                        replace_flag_1 = False
                    elif replace_flag_2:
                        print("        \"OutputFile\": \"%s\"" % new_grain_ID_PRISMS)
                        replace_flag_2  = False
                    else:
                        print("        \"OutputFile\": \"%s\"," % new_d3d)
                elif (line.find("\"Resolution\"") != -1):
                    res_flag = True
                    print(line.rstrip('\n'))
                elif (line.find("\"Dimensions\"") != -1):
                    dim_flag = True
                    print(line.rstrip('\n'))
                elif (line.find("\"x\":") != -1):
                    if (dim_flag):
                        print("            \"x\": %d," % x_elem)
                    elif (res_flag):
                        print("            \"x\": %f," % res[0])
                    else:
                        print(line.rstrip('\n'))
                elif (line.find("\"y\":") != -1):
                    if (dim_flag):
                        print("            \"y\": %d," % y_elem)
                    elif (res_flag):
                        print("            \"y\": %f," % res[1])
                    else:
                        print(line.rstrip('\n'))
                elif (line.find("\"z\":") != -1):
                    if (dim_flag):
                        print("            \"z\": %d" % z_elem)
                        dim_flag = False
                    elif (res_flag):
                        print("            \"z\": %f" % res[2])
                        res_flag = False
                    else:
                        print(line.rstrip('\n'))
                elif d3d_input_file is not None and line.find("\"InputFile\":") != -1:
                    print("        \"InputFile\": \"%s\"," % d3d_input_file)
                elif line.find("\"PeriodicBoundaries\":") != -1:
                    blah = "1" if is_periodic else "0"
                    print("        \"PeriodicBoundaries\": %s," % blah)
                else:
                    print(line.rstrip('\n'))
            else:
                if string.find(line, "CsvOutputFile=") != -1:
                    print("CsvOutputFile=" + new_csv.replace("\\", "\\\\"))
                elif string.find(line, "FeatureDataFile=") != -1:
                    print("FeatureDataFile=" + new_ftd.replace("\\", "\\\\"))
                elif string.find(line, "OutputFile=") != -1 and line.find(".dream3d") != -1:
                    print("OutputFile=" + new_d3d.replace("\\", "\\\\"))
                elif string.find(line, "OutputFile=") != -1 and line.find(".vtk") != -1:
                    print("OutputFile=" + new_vtk.replace("\\", "\\\\"))
                elif string.find(line, "OutputFile=") != -1 and line.find(".inp") != -1:
                    print("OutputFile=" + new_inp.replace("\\", "\\\\"))
                elif line.find("Dimensions\\1\\x=") != -1:
                    print("Dimensions\\1\\x=" + str(x_elem))
                elif line.find("Dimensions\\2\\y=") != -1:
                    print("Dimensions\\2\\y=" + str(y_elem))
                elif line.find("Dimensions\\3\\z=") != -1:
                    print("Dimensions\\3\\z=" + str(z_elem))
                elif line.find("Resolution\\1\\x=") != -1:
                    print("Resolution\\1\\x=" + str(res[0]))
                elif line.find("Resolution\\2\\y=") != -1:
                    print("Resolution\\2\\y=" + str(res[1]))
                elif line.find("Resolution\\3\\z=") != -1:
                    print("Resolution\\3\\z=" + str(res[2]))
                elif d3d_input_file is not None and line.find("InputFile=") == 0:
                    print("InputFile=" + d3d_input_file)
                elif line.find("PeriodicBoundaries=") == 0:
                    blah = "1" if is_periodic else "0"
                    print("PeriodicBoundaries=" + blah)
                else:
                    print(line.rstrip('\n'))
        # mute output to shell
        fnull = open(os.devnull, 'w')
        all_exist = False
        temp_csv = os.path.join(directory, csv_list[-1])
        temp_vtk = os.path.join(directory, vtk_list[-1])
        
        attempt_count = 0
        max_attempt = 10
        # raise IOError
        while not all_exist and attempt_count < max_attempt:
            subprocess.call(runner_path + " -p \"" + new_pipeline + "\" > output.txt", stdout=fnull,
                            stderr=subprocess.STDOUT, shell=True)
            # subprocess.call(runner_path + " -p \"" + new_pipeline + "\" > output.txt", shell=True)
            all_exist = os.path.exists(temp_csv) and os.path.exists(temp_vtk)
            attempt_count += 1
        if attempt_count >= max_attempt:
            raise IOError("Could not construct DREAM.3D microstructures. Verify input statistics, pipeline and DREAM.3D version")
            
    for ii in range(num_runs):
        # Rename 'orientations.txt' filename 
        try:
            os.remove(os.path.join(directory, 'orientations_%d.txt' % ii))
        except:
            pass
        os.rename(os.path.join(directory, 'orientations._%d.txt' % ii),os.path.join(directory, 'orientations_%d.txt' % ii))
    return csv_list, vtk_list

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
    filename =  os.path.join(directory, CSV_DREAM3D % num)
    
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

def read_ms_text_PRISMS(directory, num):

    """
    Read microstructure from the text file representation

    Parameters
    ----------
    directory : str or unicode
        File system path from which to read the text representation of ms
    num : int
        Microstructure number to read from the supplied folder

    Returns
    -------
    ms : ndarray
        3D array of the microstructure which was stored in the respective text file in the form of grain ID for each spatial location in the array. Reshaped from the text representation using Fortran (first fastest) ordering. Grain IDs are 0 indexed now.
    size : None or list
        list of floats for the size of the microstructure along each dimension. None if this information was not stored in the file.

    """
    ms_p = None

    elem_f_p = os.path.join(directory, PRISMS_GRAIN_ID % num)
    if not os.path.exists(elem_f_p):
        return ms_p
    with open(elem_f_p, 'r') as f:
        line = f.readline()
        shape_p = [int(s) for s in re.findall('\d+', line)][1:4]
        
    # Flip order because needs to be X, Y, Z and not Z, Y, X
    shape_p.reverse()
        
    ms_p = np.genfromtxt(elem_f_p, dtype=int, skip_header=20, skip_footer=6)
    ms_p -= 1
    
    # Need to sort element by X, then Y, then Z
    ms_p_1 = []
    for tt in range(shape_p[2]):
        for rr in range(shape_p[1]):
            for ss in range(shape_p[0]):
                ms_p_1.append( ms_p[rr + shape_p[1] * ss][tt])
                
    ms_p_2 = np.array(ms_p_1)
    ms_p_2 = np.reshape(ms_p_2, shape_p, order='F')
    
    if len(ms_p.flatten()) != shape_p[0] * shape_p[1] * shape_p[2]:
        raise ValueError('Error reading in microstructure!')
    
    return ms_p_2

def overlay_ms(ms, nodes, element_centroids, size=None, offsets=None):
    """
    Create a relationship between a finite element mesh and a microstructure representation

    Take in a 3d array microstructure, FEM nodes and elements (any kind), optional previous size, new size and offsets in each dimension numpy array
    return the element sets for each grain in the microstructure
    Any elements which are not contained in the microstructure volume are assigned a grain ID of -1 allowing them to have a different, default behavior

    Parameters
    ----------
    ms : ndarray
        3D array representing voxelated microstructure of grain ids at each index
    nodes : ndarray
        positions in 3D space used to get extents of mesh (n, 3)
    element_centroids : ndarray
        positions in 3D space used to overlay microstructure (n, 3)
    size : list, optional
        list of floats for the size of the microstructure along each axis. If not provided, assume microstructure reaches extents of the geometry
    offsets : list, optional
        list of floats locating the origin of the microstructure region.

    Returns
    -------
    grain_el : dict
        key is grain ID, values are lists of index locations to the elements array which are associated with the given grain ID
    el_grains : ndarray
        list of the grain IDs associated with each element

    """
    el_grains = np.zeros((element_centroids.shape[0]), dtype=int)
    el_grains -= 1
    max_ind = np.max(nodes, axis=0)
    shape = np.asarray(ms.shape, dtype = int)
    

    if offsets is None and size is not None:
        offsets = (max_ind - size) / 2.0
    elif size is None:
        offsets = np.zeros(len(max_ind))
        size = max_ind
       
    dim_ind = (element_centroids - offsets) / size * shape
    dim_ind = dim_ind.astype(int)
    mask_in_bounds = np.sum(np.logical_and(dim_ind > -1, dim_ind < shape), axis=1) == len(shape)
    dim_ind = dim_ind[mask_in_bounds]
    ind = tuple([dim_ind[:,i] for i in range(dim_ind.shape[1])])
    el_grains[mask_in_bounds] = ms[ind]

    num_grains = np.max(el_grains) + 1
    grain_el = {}
    for i in range(num_grains):
        grain_el[i] = np.argwhere(el_grains == i)
    if len(np.argwhere(el_grains == -1)) > 0:
        raise ValueError('Something went wrong... Check input sizes')
    return grain_el, el_grains

def make_nodes_elements(shape, size=None, mask=None):
    """
    Make a voxelated mesh of a given size and shape (elements per axis)

    Parameters
    ----------
    shape : list
        list of int for the number of elements along each axis
    size : list, optional
        list of float for the physical size in mm along each axis
    mask : ndarray, optional
        True where elements should be included (simulated in FEM model) False otherwise. mesh.shape must match shape argument

    Returns
    -------
    nodes : ndarray
        list of 3D spatial locations for the nodes in the FEM mesh (n, 3)
    elements : ndarray
        list of node connectivity for each brick elements (n, 8)

    """
    if mask is not None and mask.shape != shape:
        raise ValueError("Mask dimensions must match microstructure shape")
    nodes = []
    node_shape = (shape[0] + 1, shape[1] + 1, shape[2] + 1)
    node_map = np.zeros(node_shape)
    node_map -= 1
    index = 0
    for k in range(shape[2] + 1):
        for j in range(shape[1] + 1):
            for i in range(shape[0] + 1):
                if (mask is not None):
                    x_l = max(0, i - 1)
                    x_u = min(i + 1, shape[0])
                    y_l = max(0, j - 1)
                    y_u = min(j + 1, shape[1])
                    z_l = max(0, k - 1)
                    z_u = min(k + 1, shape[2])
                    temp = mask[x_l:x_u, y_l:y_u, z_l:z_u]
                    is_node = np.any(temp)
                else:
                    is_node = True
                if (is_node):
                    node_map[i, j, k] = index
                    n_loc = [i, j, k]
                    nodes.append(n_loc)
                    index += 1
    nodes = np.asarray(nodes, dtype=float)
    if size is not None:
        for i in range(3):
            factor = size[i] / float(shape[i])
            nodes[:,i] *= factor
    elements = []
    for k in range(shape[2]):
        for j in range(shape[1]):
            for i in range(shape[0]):
                if (mask is None or mask[i, j, k]):
                    n_list = [node_map[i, j, k], node_map[i + 1, j, k], node_map[i + 1, j + 1, k], node_map[i, j + 1, k],
                         node_map[i, j, k + 1], node_map[i + 1, j, k + 1], node_map[i + 1, j + 1, k + 1],
                         node_map[i, j + 1, k + 1]]
                    elements.append(n_list)
    elements = np.asarray(elements)
    elements = elements.astype(int)
    return nodes, elements

def get_centroids(nodes, elements):
    centroids = np.average(nodes[elements], axis=1)
    return centroids

def sub_banding_scheme(directory, i, grain_sets, number_of_layers, elem_list, el_centroids_periodic, num_vox, num_planes):
  
    """
    Iterate through each band to create unique sub-band regions

    Parameters
    ----------
    directory : str or unicode
        Path to file of interest
    i : int
        Number of instantiation
    grain_sets : list
        List of elements in each grain
    number_of_layers : ndarray
        Number of layers for each slip plane in each grain
    elem_list : dict
        List of which element belongs to which slip plane band layer
    el_centroids_periodic : ndarray
        Array of element centroids accounting for microstructure periodicity (crucial for periodic microstructure FIP volume averaging!)
    num_vox : int
        Number of desired elements in each sub-band
    num_planes : int
        Number of unique slip planes in this material system
    """    
    
    # Initialize arrays to store list of unique sub-band regions
    master_sub_band_dictionary = {}
    number_of_sub_bands = {}
    
    # Specify whether each sub-band should be written to a text file
    # This is useful for visualization purposes or to investigate the sub-band scheme, but may be set to False to prevent excessively large text files from being generated
    write_to_file = True 
    
    start_time = time.time()
    
    fname_sub = os.path.join(directory, 'sub_band_regions_%d.inp' % i)
    
    if write_to_file:
        sb_f = open(fname_sub, 'w')
    
    # Iterate through each grain
    for qq in range(len(grain_sets)):
        
        if qq % 50 == 0:
            print('Sub-banding grain: %d' % qq)
        
        # Iterate throuh each slip plane
        for rr in range(num_planes):
        
            # Iterate through the correct number of layers for the current grain and slip plane band
            for ss in range(int(number_of_layers[qq,rr])):
            
                # Iterate through elements in band
                m_el_list = []
                
                if len(elem_list[qq,rr,ss]) > num_vox:
                    # If the band in consideration has more than X elements, do this 
                    for el_in in elem_list[qq,rr,ss]:
                        temp_dist = {}
                        
                        # Iterate through all other elements in the band
                        for el_pot in elem_list[qq,rr,ss]:
                        
                            if el_in is not el_pot:
                                temp_dist[el_pot] = calc_total_dist(el_centroids_periodic[el_in-1],el_centroids_periodic[el_pot-1])
                        el_in_region_sets = get_sub_band_elem_sets(temp_dist,num_vox,el_in)
                        m_el_list = m_el_list + el_in_region_sets
                    
                    unique_el_list = [list(x) for x in set(tuple(x) for x in m_el_list)]
                    
                    for vv, item3 in enumerate(unique_el_list):
                        master_sub_band_dictionary[qq,rr,ss,vv] = item3
                        if write_to_file:
                            sb_f.write("*Elset, elset=grain_%d_plane_%d_layer_%d_subBand_%d\n" % (qq + 1, rr + 1, ss + 1,vv + 1))
                            write_set_index_elem_at_0(item3,sb_f)
                    number_of_sub_bands[qq,rr,ss] = vv + 1
                else:
                    # If band has less than num_vox elements, just make these the sub-band region
                    master_sub_band_dictionary[qq,rr,ss,0] = elem_list[qq,rr,ss]
                    if write_to_file:
                        sb_f.write("*Elset, elset=grain_%d_plane_%d_layer_%d_subBand_%d\n" % (qq + 1, rr + 1, ss + 1, 1))
                        write_set_index_elem_at_0(elem_list[qq,rr,ss],sb_f)
                    number_of_sub_bands[qq,rr,ss] = 1
                
    if write_to_file:
        sb_f.close()
    print('Sub-banding complete')
    
    # Store sub-band averaging items to pickle file
    fname = os.path.join(directory,'sub_band_info_%d.p' % i)
    h1 = open(fname,'wb')
    # Change to 'protocol = 0' to read the '.p' file in python 2.x
    p.dump([master_sub_band_dictionary,number_of_layers,number_of_sub_bands],h1, protocol = 2)
    h1.close()
    
    print("Sub-band generation time: %2.2f seconds " % (time.time() - start_time))

def calc_total_dist(el_pos,grain_pos):
    # Calculate distance between element and grain centroid (or technically any two 3d coordinates)
    # el_pos    : (x,y,z) of element centroid
    # grain_pos : (x,y,z) of grain centroid
    #
    # Can also be used to determine distance between two elements!
    if len(el_pos) != len(grain_pos):
        raise ValueError("Array length mismatch")
    dist_temp = (el_pos[0]-grain_pos[0])**2 + (el_pos[1]-grain_pos[1])**2 + (el_pos[2]-grain_pos[2])**2
    dist = np.sqrt(dist_temp)
    return dist

def get_sub_band_elem_sets(temp_distances, num_vox, el_in):
    # Calculate the set of sub-band region elements associated with an element and its neighbors in a band
    # Inputs:
    #   temp_distance : dictonary of element numbers and distances to the element in consideration
    #   num_vox       : number of elements to be assigned to a sub-band region

    if num_vox < 2:
        raise ValueError('Desired elements for sub-band region below limit')
        
    # Sort calculated distances wrt element numbers
    sorted_temp = sorted(temp_distances.items(), key=operator.itemgetter(1))
    sorted_dist = [val for item,val in sorted_temp]
    try:
        CD = sorted_dist[num_vox - 2]
    except:
        # This is a very rare error! Please disregard the current instantiation...
        raise ValueError('Number of desired voxels in sub-band region exceeds consideration elements')
        
    # Determine elements which are not degenerate
    def_less_than = [item for item,val in sorted_temp if val < CD]
    def_less_than.append(el_in)
    
    # Determine elements which are at the 'critical distance'
    CD_elems = [item for item,val in sorted_temp if val == CD]
    
    # Determine number of empty spots to fill for the element's sub-band region
    avail_spots = num_vox - len(def_less_than)
    
    # If one element is at 'critial distance', add it to the list
    if (len(CD_elems) == 1):
        el_list_of_lists_temp = def_less_than + CD_elems
        el_list_of_lists = [sorted(el_list_of_lists_temp)]
    # Otherwise, determine all combinations of elements at the same distance away from element under consideration
    else:
        el_list_of_lists = []
        if avail_spots == 1:
            for item_pot in CD_elems:
                def_less_than_temp = [item for item in def_less_than]
                def_less_than_temp.append(item_pot)
                el_list_of_lists.append(sorted(def_less_than_temp))
        elif avail_spots > 1:
            combos_pot = list(combinations(CD_elems,avail_spots))
            for item_pot in combos_pot:
                def_less_than_temp = [item for item in def_less_than]            
                for item_2 in item_pot:
                    def_less_than_temp.append(item_2)
                el_list_of_lists.append(sorted(def_less_than_temp))
        else:
            raise ValueError('avail_spots is incorrect!')
    # Return list of lists, whose length is equal to all possible element combinations for desired sub-band region
    return el_list_of_lists

def write_set_index_elem_at_0(set, o_f, exclude=[]):
    """ write a set of values respecting Abaqus limit of elements per line """
    a = 1
    for ii in range(len(set)):
        if (not set[ii] in exclude):
            if (ii == len(set) - 1):
                o_f.write("%d" % (set[ii]))
            else:
                o_f.write("%d, " % (set[ii]))
                a += 1
                if (a == 17):
                    o_f.write("\n")
                    a = 1
    o_f.write("\n")

def get_grain_centroids(el_centroids, el_grains, el_volumes):
    # Calculate centroid of each grain
    # IMPORTANT: This obviously is unclear when microstructures are generated as fully periodic because a grain that exists at "two ends of the SVE" and is split by the SVE boundary will have a grain centroid in the middle of the SVE!
    # This is not an issue for non-periodic microstructures. For periodic microstructures, this issue is resolved in the "kosher_centroids" function.
    
    """ list of element centroids (num_elements), list of element to grain number mapping (num_elements), list of element volumes (num_elements) """
    if len(el_centroids) != len(el_grains) or len(el_grains) != len(el_volumes):
        raise ValueError("Length mismatch for element arrays")
    num_grains = np.max(el_grains) + 1
    diameters = []
    centroids = np.zeros((num_grains, el_centroids.shape[1]))
    el_mask = el_grains != -1
    volumes = np.bincount(el_grains[el_mask], weights=el_volumes[el_mask])
    for i in range(len(volumes)):
        temp = volumes[i]
        temp_dia = (6. * temp / np.pi)**(1./3)
        diameters = np.append(diameters, temp_dia)
    for i in range(centroids.shape[1]):
        centroids[:,i] = np.bincount(el_grains[el_mask], weights=el_volumes[el_mask]*el_centroids[el_mask,i]) / volumes
    # return centroids, volumes, diameters
    return centroids, volumes

def write_band_sets_advanced_rev2(el_plane_layers, grain_el, directory, i):

    """
    Advanced banding scheme to absorb the first layer into the second and the last into the second-to-last
    This will remove the majority of single element bands
    
	Writes the band_sets_%d.inp file to specify which band of each grain contains which element
    
	Defintions:
    
    g_set: contains the elements for the current grain, g, in the first for loop below
    gpl  : contains which layers each element belongs to for the respective planes
	"""
    
    # If an edge band (i.e., the first or last band) contains too few elements as defined below, absorb this into the neighboring band
    # Determined by taking cube root of average number of elements per grain as defined below
    # ~100 elements per grain requires more than four elements per band      
    absorb_threshold = np.cbrt(np.shape(el_plane_layers)[0] / np.shape(grain_el)[0])

    start_time = time.time()
    
    # The code below will write the number of elements in each band to a text file
    fid = os.path.join(directory, BAND_SET_STATISTICS % i)
    f2 = open(fid, "w")    

    o_f    = os.path.join(directory, BAND_SETS_FORMAT % i)
    planes = el_plane_layers.shape[1]
    
    # Initialize array to store number of layers for each grain and slip plane
    num_layers = np.zeros((len(grain_el), planes), dtype = int)
    
    # Initialize dictionary to define what elements belong to what layers
    elem_list = {}
    
    with open(o_f, "w") as f:
        for g, g_set in enumerate(grain_el):
            if len(g_set) == 0:
                raise ValueError('Grain_el should be passed in as a list')
            
            g_set = np.asarray(g_set)
            g_set = g_set.flatten()
            gpl = el_plane_layers[g_set]
            for p in range(planes):
                pl = gpl[:,p]
                max_l = np.max(pl)
                min_l = np.min(pl)
                
                # Find layers which actually contain elements
                non_empties = []
                for l in range(max_l + 1):
                    band_els = g_set[pl == l]
                    if len(band_els):
                        non_empties.append(l)


                if (max_l - min_l > 2):

                    # If there are more than four bands in this set, consider absorbing the first band into second band and the second-to-last band into the last band
                    
                    first_layer = non_empties[0]
                    second_layer = non_empties[1]
                    second_to_last_layer = non_empties[-2]
                    last_layer = non_empties[-1]
                
                else:
                    # If grains are extremely coarse (only 1-10 elements per grain), avoid absorbing bands
                    
                    first_layer = []
                    second_layer = []
                    second_to_last_layer = []
                    last_layer = []

                
                for l in range(max_l + 1):
                    band_els = g_set[pl == l]
                    
                    # If the first or last band has too few elements, absorb it into neighboring band
                    if (l in [first_layer, second_layer]) and (len(g_set[pl == first_layer]) < absorb_threshold):
                    
                        absorb_first_outer_bands = True
                        absorb_last_outer_bands  = False
                        
                    elif (l in [second_to_last_layer, last_layer]) and (len(g_set[pl == last_layer]) < absorb_threshold):
                    
                        absorb_first_outer_bands = False
                        absorb_last_outer_bands  = True
                    else:
                    
                        absorb_first_outer_bands = False
                        absorb_last_outer_bands  = False



                    if (max_l - min_l > 2) and (len(non_empties) > 3) and (absorb_first_outer_bands or absorb_last_outer_bands):
                        if (l == first_layer) and absorb_first_outer_bands:
                            # Absorb first band into second band
                            temp3 = band_els
                            
                        elif (l == second_layer) and absorb_first_outer_bands:
                            band_els = np.concatenate((band_els,temp3))
                            if len(band_els):
                                # Write element numbers to abaqus input file
                                f.write("*Elset, elset=grain_%d_plane_%d_layer_%d\n" % (g + 1, p + 1, l + 1))
                                write_set(band_els, f)
                                
                                # Write number of elements in this band
                                f2.write("%d   %d   %d   %d\n" % (g+1, p+1, num_layers[g,p]+1, len(band_els)))

                                # Keep internal track of element list for sub-banding scheme
                                elem_list[g,p,num_layers[g,p]] = [int(yy)+1 for yy in band_els]
                                num_layers[g,p] += 1
                        
                        elif (l == second_to_last_layer) and absorb_last_outer_bands:
                            # Absorb last band into second-to-last band
                            temp4 = band_els
                            
                        elif (l == last_layer) and absorb_last_outer_bands:
                            band_els = np.concatenate((band_els,temp4))
                            if len(band_els):
                                f.write("*Elset, elset=grain_%d_plane_%d_layer_%d\n" % (g + 1, p + 1, l + 1))
                                write_set(band_els, f)
                                
                                f2.write("%d   %d   %d   %d\n" % (g+1, p+1, num_layers[g,p]+1, len(band_els)))
                                
                                elem_list[g,p,num_layers[g,p]] = [int(yy)+1 for yy in band_els]
                                num_layers[g,p] += 1
                                
                        else:
                            # Leave all other bands unchanged
                            if len(band_els):
                                f.write("*Elset, elset=grain_%d_plane_%d_layer_%d\n" % (g + 1, p + 1, l + 1))
                                write_set(band_els, f)
                                
                                f2.write("%d   %d   %d   %d\n" % (g+1, p+1, num_layers[g,p]+1, len(band_els)))
                                
                                elem_list[g,p,num_layers[g,p]] = [int(yy)+1 for yy in band_els]
                                num_layers[g,p] += 1
                    else:
                        # If there are less than four bands in this set, don't absord bands together
                        if len(band_els):
                            # print('Less than four bands at location' , g + 1, p + 1, l + 1)
                            f.write("*Elset, elset=grain_%d_plane_%d_layer_%d\n" % (g + 1, p + 1, l + 1))
                            write_set(band_els, f)
                            
                            f2.write("%d   %d   %d   %d\n" % (g+1, p+1, num_layers[g,p]+1, len(band_els)))
                            
                            elem_list[g,p,num_layers[g,p]] = [int(yy)+1 for yy in band_els]
                            num_layers[g,p] += 1
                          
                          
        f.write("*")

    print('Amount of bands in total is %s' % np.sum(num_layers))
    f2.close()
        
    print("Time to band microstructure: %2.2f seconds " % (time.time() - start_time))
    
    return num_layers, elem_list

def write_set(set, o_f, exclude=[]):
    """ write a set of values respecting Abaqus limit of elements per line """
    a = 1
    for ii in range(len(set)):
        if (not set[ii] in exclude):
            if (ii == len(set) - 1):
                o_f.write("%d" % (set[ii] + 1))
            else:
                o_f.write("%d, " % (set[ii] + 1))
                a += 1
                if (a == 17):
                    o_f.write("\n")
                    a = 1
    o_f.write("\n")

def list_elem_in_band(grain,plane,input_doc, kk):

    # Deprecated and unused function, but included for user reference 
    
    global layer_counter
    global number_of_total_bands

    line_num = []
    lookup = '*Elset, elset=grain_%s_plane_%s' % (grain, plane)
    number_of_layers = 0
    with open(input_doc) as fid:
        for num, line in enumerate(fid, 1):
            if lookup in line:
                line_num.append(num)
                number_of_layers += 1			
    inRange = True
    index = 1
    clean_list = []
    band_list = []
    if len(line_num) == 0: #Band does not exist
        print('Band does not exist')
        band_list.append(0)
    else:
        while inRange:
            #Test if line contains element numbers
            if '*' in linecache.getline(input_doc, (line_num[layer_counter] + index)):
                inRange = False
                break
            #Add elements in line to list of elements
            list = (linecache.getline(input_doc, (line_num[layer_counter] + index))).split(' ')
            #clean the list and change it to integers
            for item in list:
                clean_list.append((item.replace("\n", "")).replace(",", ""))
            while "" in clean_list: clean_list.remove("")

            band_list_temp = [int(yy) for yy in clean_list]

            band_list.append(band_list_temp)
            index = index + 1
    layer_counter += 1
    return band_list_temp, len(band_list_temp)

def get_num_layers(grain,plane,input_doc):

    # Deprecated and unused function, but included for user reference 

    lookup = '*Elset, elset=grain_%s_plane_%s' % (grain, plane)
    number_of_layers = 0
    with open(input_doc) as fid:
        for num, line in enumerate(fid, 1):
            if lookup in line:
                number_of_layers += 1
    return number_of_layers	

def kosher_centroids(el_cen, g_cen, size, grain_sets, el_grain, el_volumes):
    # Shift elements so that grain centroids are not incorrectly "in the middle" of the SVE for grains split by an SVE boundary/face
    
    # Added functionality to check whether a grain is flat (in the shape of a pancake)
    # If so, do not attempt to move elements in this direction as it will just oscillate back and forth and crash!

    # Calculate realistic grain centroids by enforcing element periodicity
    a = size[0]
    b = size[1]
    c = size[2]
    contt = True
    x_flag = False
    y_flag = False
    z_flag = False
    
    # Create copy of element centroids which will be overwritten to reflect 'periodicity'
    hyp_cen = el_cen
    print('Calculate realistic grain centroids')
    print('')
    for k in range(len(g_cen)):
        singl_elem_z = False
        singl_elem_y = False
        singl_elem_x = False
        if k % 50 == 0:
            print('Working on grain %d' % k)
        contt = True
        iter_count = 0
        while contt:
            iter_count += 1
            x_flag = False
            y_flag = False
            z_flag = False
            if (iter_count % 1000) == 0:
                print('Periodic grain iteration: %d' % iter_count)
            if iter_count > 100:
                raise ValueError('Max iterations reached; something went wrong... please try again!')
                
                
            
            # Use to find flat, "pancake" shaped grains
            tt1_temp = el_cen[np.where(el_grain==k)]    
                
            # *********************** X direction ******************************************************************************************
            if not np.all(tt1_temp == tt1_temp[0,:], axis=0)[0]:
                
                # Determine which element is farthest away and to which grain it belongs
                max_dist_x, keyy_x = calc_directional_max_dist_elem_grain(el_cen,g_cen[k],grain_sets[k],0)

                # Shift element(s) in the X direction by SVE size in X direction
                for rr, keyy_x_iter in enumerate(keyy_x):
                    if g_cen[k][0] >= el_cen[keyy_x_iter][0]:
                        hyp_cen[keyy_x_iter][0] += a
                    else:
                        hyp_cen[keyy_x_iter][0] -= a
                        
                # Calculate hypothetical grain center if element is moved in X direction
                g_cen_hyp, volumes = calc_hyp_grain_centroid(hyp_cen, el_grain, el_volumes)
                
                # Calculate hypothetical distance in the X direction between shifted element and grain centroid
                dist_x_hyp = calc_single_dist(hyp_cen[keyy_x_iter][0],g_cen_hyp[k][0])

                if dist_x_hyp <= max_dist_x:
                    el_cen = hyp_cen
                    g_cen = g_cen_hyp
                    x_flag = False
                else:
                    for rr, keyy_x_iter in enumerate(keyy_x):
                        if g_cen[k][0] >= el_cen[keyy_x_iter][0]:
                            hyp_cen[keyy_x_iter][0] += a
                        else:
                            hyp_cen[keyy_x_iter][0] -= a
                    x_flag = True
                    # print 'x flag true'
            else:
                x_flag = True
                singl_elem_x = True
                
            # *********************** Y direction ******************************************************************************************            
            if not np.all(tt1_temp == tt1_temp[0,:], axis=0)[1]:
            
                # Determine which element is farthest away and to which grain it belongs
                max_dist_y, keyy_y = calc_directional_max_dist_elem_grain(el_cen,g_cen[k],grain_sets[k],1)
                
                # Shift element(s) in the Y direction by SVE size in Y direction
                for ss, keyy_y_iter in enumerate(keyy_y):
                    if g_cen[k][1] >= el_cen[keyy_y_iter][1]:
                        hyp_cen[keyy_y_iter][1] += b
                    else:
                        hyp_cen[keyy_y_iter][1] -= b
                        
                # Calculate hypothetical grain center if element is moved in X direction
                g_cen_hyp, volumes = calc_hyp_grain_centroid(hyp_cen, el_grain, el_volumes)
                
                # Calculate hypothetical distance in the X direction between shifted element and grain centroid
                dist_y_hyp = calc_single_dist(hyp_cen[keyy_y_iter][1],g_cen_hyp[k][1])
                
                if dist_y_hyp <= max_dist_y:
                    el_cen = hyp_cen
                    g_cen = g_cen_hyp
                    y_flag = False
                else:
                    for ss, keyy_y_iter in enumerate(keyy_y):            
                        if g_cen[k][1] >= el_cen[keyy_y_iter][1]:
                            hyp_cen[keyy_y_iter][1] += b
                        else:
                            hyp_cen[keyy_y_iter][1] -= b
                    y_flag = True  
                    # print 'y flag true'
            else:
                y_flag = True
                singl_elem_y = True
                
            # *********************** Z direction ******************************************************************************************  
            if not np.all(tt1_temp == tt1_temp[0,:], axis=0)[2]:

                # Determine which element is farthest away and to which grain it belongs
                max_dist_z, keyy_z = calc_directional_max_dist_elem_grain(el_cen,g_cen[k],grain_sets[k],2)     
                
                # Shift element(s) in the Z direction by SVE size in Z direction
                for tt, keyy_z_iter in enumerate(keyy_z):                
                    if g_cen[k][2] >= el_cen[keyy_z_iter][2]:
                        hyp_cen[keyy_z_iter][2] += c
                    else:
                        hyp_cen[keyy_z_iter][2] -= c
                        
                # Calculate hypothetical grain center if element is moved in X direction
                g_cen_hyp, volumes = calc_hyp_grain_centroid(hyp_cen, el_grain, el_volumes)
                
                # Calculate hypothetical distance in the X direction between shifted element and grain centroid
                dist_z_hyp = calc_single_dist(hyp_cen[keyy_z_iter][2],g_cen_hyp[k][2])
                
                if dist_z_hyp <= max_dist_z:
                    el_cen = hyp_cen
                    g_cen = g_cen_hyp
                    z_flag = False
                else:
                    for tt, keyy_z_iter in enumerate(keyy_z):               
                        if g_cen[k][2] >= el_cen[keyy_z_iter][2]:
                            hyp_cen[keyy_z_iter][2] += c
                        else:
                            hyp_cen[keyy_z_iter][2] -= c
                    z_flag = True
                    # print 'z flag true'   
            else:
                z_flag = True
                singl_elem_z = True
            
            # if singl_elem_z and singl_elem_y and singl_elem_x:
                # print('Just kidding. Grain %d is a single element.' % (k+1))
            if (x_flag == True) and (y_flag == True) and (z_flag == True):
                contt = False
                #print 'iterations for grain %d: %d' % (k+1, iter_count)
    # return element and grain centroids which reflect periodicity!   
    return el_cen, g_cen

def calc_directional_max_dist_elem_grain(el_cen,g_cen,g_set_elem,dir):
    # Calculate distance and ID of element farthest away from grain centroid in direction dir
    # el_cen     : (x,y,z) list of element centroid in one direction
    # g_cen      : (x,y,z) centroid of grain in one direction
    # g_set_elem : list of element belonging to grain g_cen
    # dir (int)  : direction under consideration (0=x, 1=y, 2=z)
    #
    # Outputs:
    # max_dist   : max distance between grain centroid and element
    # key        : list of element IDs which are farthest from grain centroid
    max_dist = -1
    key = []
    for i, elem_g in enumerate(g_set_elem):
        temp_dist = calc_single_dist(el_cen[elem_g[0]][dir],g_cen[dir])
        # temp_dist = calc_total_dist(el_cen[elem_g[0]],g_cen)
        # print i, elem_g
        if temp_dist > max_dist:
            # print 'distance updated'
            max_dist = temp_dist
            key = [elem_g[0]]
        elif temp_dist == max_dist:
            key.append(elem_g[0])
    return max_dist, key
    
def calc_single_dist(el_pos,grain_pos):
    # Calculate distance between element and grain centroid in a single direction
    # el_pos    : SINGLE coordinate position of element (x, y or z)
    # grain_pos : SINGLE coordinate position of grain (x, y or z)
    dist = np.sqrt( (el_pos-grain_pos)**2 )
    return dist

def calc_hyp_grain_centroid(el_centroids, el_grains, el_volumes):
    """ list of element centroids (num_elements), list of element to grain number mapping (num_elements), list of element volumes (num_elements) """
    if len(el_centroids) != len(el_grains) or len(el_grains) != len(el_volumes):
        raise ValueError("Length mismatch for element arrays")
    num_grains = np.max(el_grains) + 1
    centroids = np.zeros((num_grains, el_centroids.shape[1]))
    el_mask = el_grains != -1
    volumes = np.bincount(el_grains[el_mask], weights=el_volumes[el_mask])
    for i in range(centroids.shape[1]):
        centroids[:,i] = np.bincount(el_grains[el_mask], weights=el_volumes[el_mask]*el_centroids[el_mask,i]) / volumes
    return centroids, volumes

def make_banded(el_centroids, el_grain, grain_centroids, orientations, planes, band_width, directory, i):
    # Determine "bands" corresponding to slip planes based on each grain's crystallographic orientation
    # Addition below to create text file with band plane normals
    fid = os.path.join(directory, BAND_PLANE_NORMALS % i)
    f = open(fid, "w")

    el_plane_layers = np.zeros((el_centroids.shape[0], planes.shape[0]))
    num_grains = np.max(el_grain) + 1
    for g in range(num_grains):
        ori = orientations[g]
        g_mask = el_grain == g
        temp_dists = el_centroids[g_mask] - grain_centroids[g]
        g_els = np.where(g_mask)
        for i, p in enumerate(planes):
            plane_norm = bunge_euler_rotation(ori, p)
            f.write("%1.15f  %1.15f  %1.15f\n" % (plane_norm[0], plane_norm[1], plane_norm[2]))
            el_plane_layers[g_els,i] = np.tensordot(temp_dists, plane_norm, axes=[1,0]) / band_width
    el_plane_layers = np.ceil(el_plane_layers)
    el_plane_layers -= np.min(el_plane_layers)
    el_plane_layers = el_plane_layers.astype(int)
    f.close()
    return el_plane_layers

def bunge_euler_rotation(orientation, vector):
    # Although the rot_matrix indices are reversed, this is the proper way to rotate the slip plane normal vectors for the banding process

    s1 = np.sin(orientation[0])
    s2 = np.sin(orientation[1])
    s3 = np.sin(orientation[2])
    c1 = np.cos(orientation[0])
    c2 = np.cos(orientation[1])
    c3 = np.cos(orientation[2])
    rot_matrix = np.zeros((3, 3))
    rot_matrix[0, 0] = c1 * c3 - s1 * s3 * c2
    rot_matrix[0, 1] = s1 * c3 + c1 * s3 * c2
    rot_matrix[0, 2] = s3 * s2
    rot_matrix[1, 0] = -c1 * s3 - s1 * c3 * c2
    rot_matrix[1, 1] = -s1 * s3 + c1 * c3 * c2
    rot_matrix[1, 2] = c3 * s2
    rot_matrix[2, 0] = s1 * s2
    rot_matrix[2, 1] = -c1 * s2
    rot_matrix[2, 2] = c2
    return np.dot(vector, rot_matrix)

def convert_dict_lists(grain_el):
    num_grains = max(grain_el.keys()) + 1
    new_grain_el = [ grain_el[i] for i in range(num_grains) ]
    el_els = []
    if -1 in grain_el:
        el_els = grain_el[-1]
    return new_grain_el, el_els

def get_top_bottom_face_el(el_cen, FD, ms_list, jj):
    """
    Function to determine which grains are split by a single non-periodic SVE face/boundary
    
    Parameters
    ----------
    el_cen : array
        Element centroids
    FD : 0, 1 or 2
        Direction which is set to non-periodic, corresponds to X, Y, or Z. Typically, Y direction will be set to non-periodic in simulations.
    ms_list : list 
        Flattened list of which element belongs to each grain
    jj : int
        Instantiation number

    Returns
    -------
    sorted_sg_temp : list
        Sorted ist of grains which are split by the non-periodic face
   
    """
    
    # Determine location of top and bottom of non-periodic direction of interest
    el_cen_t = el_cen.transpose()
    maxx = el_cen_t[FD].max()
    minn = el_cen_t[FD].min()
    bottom_loc = [i for i, j in enumerate(el_cen_t[FD]) if j == minn]
    top_loc = [i for i, j in enumerate(el_cen_t[FD]) if j == maxx]
    
    ms_bot = ms_list[bottom_loc]
    ms_top = ms_list[top_loc]
    
    # Grains at the bottom of direction of interest
    output_bot = []
    for x in ms_bot:
        if x not in output_bot:
            output_bot.append(x)
            
    output_bot.sort()
    
    # Grains at the top of direction of interest    
    output_top = []
    for x in ms_top:
        if x not in output_top:
            output_top.append(x)
    
    output_top.sort()
    
    # Find grains at both top and bottom of boundary split
    split_grains = set(output_bot) & set(output_top)
    sorted_sg_temp = sorted(split_grains)
    sorted_sg = [x+1 for x in sorted_sg_temp]
    
    # Write to text file
    with open('split_grains_%d.txt' % jj, 'w') as f:
        f.write('Grains split by the non-periodic face, indexed at 1\n')
        for item in sorted_sg:
            f.write("%d\n" % item)
            
    return sorted_sg_temp

def elem_reindex_split_face_plane(el_in_grain, el_centroids, FD):
    """
    Function to determine which elements will be indexed as a new grain due to splitting by non-periodic face
    This function is evaluated ONCE per grain set
    
    Parameters
    ----------
    el_in_grain : array
        Elements which belong to the grain of interest
    el_centroids :
        Element centroids
    FD : 0, 1 or 2
        Direction which is set to non-periodic, corresponds to X, Y, or Z. Typically, Y direction will be set to non-periodic in simulations.

    Returns
    -------
    el_in_grain : list
        List of elements which are NOT to be indexed (subtract out el_renum)
    el_renum_arrayed : list
        List of elements to be reindex as a new grain
    """
    el_cen_t = el_centroids.transpose()    
    el_renum = []
    
    # Round before comparison is made to avoid floating point error!
    el_cen_t = np.around(el_cen_t,8)
    
    plane_flag = True
    
    # Calculate element size
    elem_size = round(el_cen_t[0][-1] - el_cen_t[0][-2],8)
    
    # Get max position of the non-periodic Face, specified by FD
    # The variable is "y_plane" but does, in fact, correspond to whichever direction is set to non-periodic
    y_plane = el_cen_t[FD][:][-1]
    count_me = 1
    
    while plane_flag:
        
        # Round before comparison is made to avoid floating point error!
        y_plane = round(y_plane,8)
        
        # Determine all elements on the positive Y-Face (first layer of elements in the Y plane) or which-ever direction is non-periodic  
        
        el_in_y_plane = np.where(el_cen_t[FD] == y_plane)[0]
        t1 = set(el_in_grain.flatten()) & set(el_in_y_plane)
        t3 = [x for x in t1]
        
        # If elements in the plane belong to the current grain of interest, sort accordingly
        if t1:
            el_renum.append(t3)
            count_me += 1
            # Iterate backward of first array below because it decreases by the element size in the X direction
            # ASSUMES CUBIC SHAPE AND SIZE OF VOXELS!
            y_plane -= elem_size

        else:
            plane_flag = False
    
    # Flatten, sort and manipulate as necessary
    el_renum_flat = [x for l in el_renum for x in l]
    el_renum_flat.sort()
    el_renum_arrayed = []
    for items2 in el_renum_flat:
        el_renum_arrayed.append([items2])    
    
    # Delete the list of elements to be indexed from the split grain
    el_in_grain = np.delete(el_in_grain, np.where(el_in_grain == el_renum_flat)[0], 0)

    return el_in_grain, el_renum_arrayed

def store_grains(directory, grain_sets, num):
    # Store which elements belong to each grain for simpler FIP averaging over entire grains
    fname5 = os.path.join(directory, 'element_grain_sets_%d.p' % num)
    h5 = open(fname5,'wb')
    p.dump(grain_sets, h5)
    h5.close()
 
def store_bands(directory, elem_list_2, num_layers, num):
    # Store which elements belong to each grain for simpler FIP averaging over entire bands
    fname5 = os.path.join(directory, 'element_band_sets_%d.p' % num)
    h5 = open(fname5,'wb')
    p.dump([elem_list_2, num_layers], h5)
    h5.close()

def append_bands_to_vtk(directory, elem_list_2, num_layers, grain_sets, shape, num_planes, num):
    # For visualization purposes of the banding process, append the bands to the .vtk file

    num_grains = len(grain_sets)
    
    # Initialize array to track to which band each element belongs
    elem_bands = np.zeros((np.product(shape), 4), dtype = int)
    
    # Iterate over grains
    for ii in range(num_grains):
        
        # Iterate over planes
        for jj in range(num_planes):
        
            # Iterate through layers
            for kk in range(num_layers[ii][jj]):
        
                for ll in elem_list_2[ii,jj,kk]:
                
                    elem_bands[ll-1].transpose()[jj] = kk


    elem_bands_reshaped = np.reshape(elem_bands,(shape[0] * shape[1] * shape[2] * num_planes), order = 'C')

    # Specify file names to visualize bands
    Fname_vtk_loc = os.path.join(os.getcwd(), 'Output_FakeMatl_%d.vtk' % num)
    Fname_vtk_new = os.path.join(os.getcwd(), 'Output_FakeMatl_%d_bands.vtk' % num)

    # Create copy of original .vtk file in case something goes wrong!
    shutil.copy(Fname_vtk_loc, Fname_vtk_new)

    # Open and write to .vtk
    f_vtk = open(Fname_vtk_new,'a')
    f_vtk.write('SCALARS Bands float 4\n')
    f_vtk.write('LOOKUP_TABLE default\n')

    # Write to .vtk
    counter = 0
    for kk in elem_bands_reshaped:
        f_vtk.write(' %g' % kk)
        counter += 1 
        if counter == 20:
            f_vtk.write('\n')
            counter = 0

    f_vtk.close()

def print_params(directory, size, shape, face_bc, num_vox, band_thickness, num_planes, num_instantiations, d3d_input_file):
    # Print current microstructure info to text file
    
    # Create/go to desired directory
    if os.path.exists(directory):
        os.chdir(directory)
    else:
        os.makedirs(directory)
        os.chdir(directory)
    
    fname = os.path.join(directory, 'microstructure_parameters.txt')
    f = open(fname, 'w')
    f.write('*** Parameters for the microstructures in this simulation folder ***\n\n')
    f.write('Directory:  %s \n' % directory)
    f.write('DREAM.3D Input file:  %s \n' % os.path.basename(d3d_input_file))
    f.write('Size (mm) in the X, Y, and Z:  %0.4f, %0.4f, %0.4f\n' % (size[0], size[1], size[2]))
    f.write('Shape in the X, Y, and Z:  %d, %d, %d\n' % (shape[0], shape[1], shape[2]))
    f.write('Elements per sub-band:  %d \n' % num_vox)
    f.write('Number of element band thickness:  %d \n' % band_thickness)
    f.write('Number of slip planes for banding:  %d \n' % num_planes)
    f.write('Face boundary conditions in X, Y, and Z:  %s, %s, %s \n' % (face_bc[0], face_bc[1], face_bc[2]))
    f.write('Number of microstructure instantiations generated:  %d' % num_instantiations)
    f.close()

def gen_microstructures(directory, size, shape, face_bc, num_vox, band_thickness, num_planes, create_sub_bands, num_instantiations, generate_new_microstructure_files, d3d_input_file, d3d_pipeline_path, d3d_executable_path): 
    # Main function
    
    # Create/go to desired directory
    if os.path.exists(directory):
        os.chdir(directory)
    else:
        os.makedirs(directory)
        os.chdir(directory)

    # Start timer
    start_time = time.time()

    # If all faces are set to "free", then generate microstructure(s) without periodicity
    if face_bc[0] == face_bc[1] == face_bc[2] == 'free':
        is_periodic = False
    else:
        # Otherwise, generate a fully periodic microstructure
        is_periodic = True

    # Generate microstructure instantiation(s) using DREAM.3D
    # If generate_new_microstructure_files is set to false, new microstructures will NOT be generated and the script will proceed with analysis of existing microstructures by reading the .csv and GrainID_#.txt files.
    if generate_new_microstructure_files:
        batch_dream3d(d3d_executable_path, d3d_pipeline_path, d3d_input_file, directory, shape, size, num_instantiations, is_periodic)
    
    # Read in plane normal directions
    planes = FCC_OCT_NORMALS
    
    # Set band width as one element thick; elements must be cubic in shape!
    if np.round(size[0] / shape[0], 12) == np.round(size[1] / shape[1], 12) == np.round(size[2] / shape[2], 12):
        # Need to round due to precision
        band_width = band_thickness * np.round(size[0] / shape[0], 12)
        
        print('Band width of %g' % band_width)
        
        # Quick check to make sure band_width makes sense
        if np.round(band_width * shape[0],8) / band_thickness != size[0]:
            raise IOError('Please fix rounding!')
    else:
        raise ValueError('Elements are not cubic in shape!')

    
    # Iterate through all microstructures to generate files necessary to volume average Fatigue Indicator Parameter (FIP) for Extreme Value Distributions (EVDs)
    for num in range(num_instantiations):
        # Create nodes and elements
        nodes, elements = make_nodes_elements(shape, size)
        
        # Get element centroids
        el_centroids = get_centroids(nodes, elements)
        
        if num == 0:
            # Store the unaltered element centroids to plot highest FIPs as a function of distance to free surface
            # Only one of these is needed for a batch simulation folder since all the element centroids will be the same across instantiations
            write_el_centroids_pickled(el_centroids, directory, num)

        # Calculate element volumes
        single_vol = np.prod(np.asarray(size)/np.asarray(shape))
        el_volumes = np.asarray([single_vol]*(elements.shape[0]))    
    
        # Read in which element belongs to which grain
        ms_list = read_ms_text_PRISMS(directory, num)

        # Read in phases, diameters, and orientations from D
        phases, diameters, orientations = read_d3d_csv(directory, num)

        # Overlay information on mesh
        grain_sets, el_grain = overlay_ms(ms_list, nodes, el_centroids, size)
        if type(grain_sets) == type({}):
            grain_sets, el_els = convert_dict_lists(grain_sets)
            
        # Store which elements belong to each grain for simpler FIP averaging over entire grains
        store_grains(directory, grain_sets, num)        

        # Determine if any surfaces are set to 'free'
        free_surface = [tt for tt, qq in enumerate(face_bc) if qq in 'free']

        # If a single direction/set of parallel faces are set to non-periodic, then grains will be periodic across this boundary as generated by DREAM.3D and must be split for proper FIP averaging
        # Split these grains and index them as new grains
        # Only perform on fully-periodic microstructures from DREAM.3D so that periodicity is preserved in the other two directions
        # This will not at all affect the simulations in PRISMS-Plasticity, but rather the volumes over which FIPs are averaged
        
        if (face_bc.count('free') == 1) and not (face_bc[0] == face_bc[1] == face_bc[2]):
            print('Splitting grains across SVE boundary')
            # Determine which grains are split by the non-periodic face
            # ONLY do this if a single direction is set to non-periodic (non-periodicity supported in one direction)
            split_g = get_top_bottom_face_el(el_centroids, free_surface[0], ms_list.flatten(order='F'), num)
            
            for kk in split_g:
                # Iterate through all grains identified as split by the non-periodic SVE boundary
                # Identify element at the top of the X, Y, or Z-Face to be indexed as new grains, edit grain_sets by removing those elements
                # ***** ASSUMES CUBIC SHAPE AND SIZE OF ELEMENTS *****
                grain_sets[kk], el_renum = elem_reindex_split_face_plane(grain_sets[kk], el_centroids, free_surface[0])
                
                # Add new indexed grains to grain_sets array
                grain_sets.append(np.asarray(el_renum))
                
                # Edit el_grain; flatten list, then overwrite el_grain
                el_renum_f = [item for sublist in el_renum for item in sublist]
                el_grain[el_renum_f] = len(grain_sets) - 1

                # Edit phase array; assign phase of new grain to be that of the grain it is split from
                phases = np.append(phases, phases[kk])
            
                # Edit orientation array
                orientations = np.vstack((orientations,orientations[kk]))

        # Calculate grain centroids
        grain_centroids, grain_volumes = get_grain_centroids(el_centroids, el_grain, el_volumes)
        
        if (face_bc.count('free') == 1) or (face_bc[0] == face_bc[1] == face_bc[2] == 'periodic'):
            # Calculate "Kosher" grain centroids, which take into account DREAM.3D creating periodic microstructures where the same grain ID can exist on opposites of an SVE, SPLIT by the SVE boundary
            el_centroids_periodic, grain_centroids_periodic = kosher_centroids(el_centroids,grain_centroids,size,grain_sets,el_grain,el_volumes)
        
            # Calculate layers of bands
            el_plane_layers = make_banded(el_centroids_periodic, el_grain, grain_centroids_periodic, orientations, planes, band_width, directory, num)
            
            # Write bands to text file
            num_layers, elem_list_2 = write_band_sets_advanced_rev2(el_plane_layers, grain_sets, directory, num)

            # Store which elements belong to each band for FIP averaging over bands
            store_bands(directory, elem_list_2, num_layers, num)
            
            if create_sub_bands:
                # Create sub-band regions and files necessary for FIP volume averaging
                sub_banding_scheme(directory, num, grain_sets, num_layers, elem_list_2, el_centroids_periodic, num_vox, num_planes)
            
        elif (face_bc[0] == face_bc[1] == face_bc[2] == 'free'):    
            # Calculate layers of bands
            el_plane_layers = make_banded(el_centroids, el_grain, grain_centroids, orientations, planes, band_width, directory, num)

            # Write bands to text file
            num_layers, elem_list_2 = write_band_sets_advanced_rev2(el_plane_layers, grain_sets, directory, num)
            
            # Store which elements belong to each band for FIP averaging over bands
            store_bands(directory, elem_list_2, num_layers, num)
            
            if create_sub_bands:
                # Create sub-band regions and files necessary for FIP volume averaging
                sub_banding_scheme(directory, num, grain_sets, num_layers, elem_list_2, el_centroids, num_vox, num_planes)
        else:
            raise ValueError('Please fix types of face boundary conditions!')
            
        append_bands_to_vtk(directory, elem_list_2, num_layers, grain_sets, shape, num_planes, num)
        
    print("Total time to generate microstructures: %2.2f seconds " % (time.time() - start_time))

def main():
    # Run from command prompt 
    
    
    ''' Specify directories and paths '''
    # Directory where microstructure data should be generated and pre-processed
    # This command creates a directory in the same location as the "PRISMS-Fatigue" directory with python scripts and DREAM.3D files 
    directory = os.path.dirname(DIR_LOC) + '\\tutorial\\test_run_1'
    
    # Alternatively, the directory can be expressed as an absolute path as: 
    # directory = r'C:\Users\stopk\Documents\GitHub\PRISMS-Fatigue\tutorial\test_run_1'
    
    # Location of DREAM.3D input file; should consist of only the "StatsGenerator" and "Write DREAM.3D Data File"
    # "StatsGenerator" inputs include grain size distribution, crystallographic texture, grain morphology, etc.
    # Six ".dream3d" files are included in PRISMS-Fatigue
    d3d_input_file = os.path.abspath(DIR_LOC) + '\\Al7075_random_texture_equiaxed_grains.dream3d'
    
    # Once again, this may be specified using an absolute path as:
    # d3d_input_file = r'C:\Users\stopk\Documents\GitHub\PRISMS-Fatigue\Al7075_cubic_texture_equiaxed_grains.dream3d'

    # Average grain size as determined in the "StatsGenerator" filter of the .dream3d file above
    # Used for automated band and sub-band sizing as shown below but which can be overwritten by the user
    avg_grain_size = 0.014 # millimeters
    
    # Location of DREAM.3D .json pipeline 
    # This can be modified by the user to include additional outputs
    d3d_pipeline_path = os.path.abspath(DIR_LOC) + '\\Dream3D_microstructure_pipeline.json'
    
    # Once again, this may be specified using an absolute path as:
    # d3d_pipeline_path = r'C:\Users\stopk\Documents\GitHub\PRISMS-Fatigue\Dream3D_microstructure_pipeline.json'
    
    # Location of DREAM.3D 'PipelineRunner.exe' file; this should be in the DREAM.3D program folder
    d3d_executable_path = r'C:\Users\stopk\Desktop\DREAM3D-6.5.141-Win64\PipelineRunner.exe'
    
    
    ''' Specify desired microstructure size and shape '''
    # Size of microstructure instantiations in millimeters, in the X, Y, and Z directions, respectively.
    size  = np.asarray([.0725,.0725,.0725])
    
    # Shape of microstructure instantiations (number of voxels/elements), in the X, Y, and Z directions, respectively.
    # IMPORTANT: at this point, only CUBIC voxel functionality supported even with a non-cubic microstructure
    # I.e., size = [.05, .1, .025] and shape = [50, 100, 25] is acceptable
    shape = np.asarray([29,29,29])  
    
    # Number of microstructure instantiations to generate using DREAM.3D
    num_instantiations = 5
    
    
    ''' Specify details of banding and sub-banding process '''
    # Specify the number of elements in each sub-band for volume averaging
    # Please see the references below for more information
    # WARNING!: If this is too low for very refined grains, this module will take a long time to run!
    # This is because it attempts to determine all UNIQUE combinations of some number of neighboring elements
    # NOTE: Aim for ~8-10% of the average grain volume as the num_vox
    
    # This line calculates num_vox to be 10% of the predicted average number of elements per grain
    num_vox = np.around(np.prod(shape) / (np.prod(size) / ( (1/6) * np.pi * avg_grain_size ** 3  ) ) * 0.10).astype(int)
    num_vox = np.around(np.prod(shape) / (np.prod(size) / ( (1.0/6.0) * np.pi * avg_grain_size ** 3  ) ) * 0.10).astype(int)
    
    # Comment out the above line and uncomment the line below to manually set the number of elements per sub-band
    # num_vox = 8

    # Specify the thickness of bands in terms of number of elements
    # Ideally, this should result in approximately 6 bands for coarser grains (~100 elements per grain)
    # NOTE: This should also be ~one or a few micrometers in thickness based on experimental observations of slip bands
    # See reference below, Castelluccio and McDowell
    
    # This line calculates the band thickness IN MULTIPLES OF element thickness 
    band_thickness = np.around( avg_grain_size / ( 6 * size[0]/shape[0]) ).astype(int)
    
    # Comment out the above line and uncomment the line below to manually set the band element thickness
    # band_thickness = 2

    # Number of crystallographic slip planes for FIP Volume averaging
    # There are four slip planes in the Al 7075-T6 material system (see references below)
    num_planes = 4
    
    # Specify whether bands in grains should be further assigned to unique sub-bands
    # WARNING!: this is slow for a) microstructures with many grains, and 2) very fine grains, i.e., thousands of elements per grain
    # NOTE: Microstructure(s) can be created with this intially set to False, and the "gen_microstructures" can be executed with the variable "generate_new_microstructure_files" set to False, so that the SAME microstructure(s) can undergo the sub-banding process at a later time.
    create_sub_bands = True


    ''' Specify boundary conditions '''
    # Boundary conditions, either "periodic" or "free surface", for X,Y,Z directions
    # Three possible combinations: 1) all 'periodic', 2) all 'free', or 3) two 'periodic' + one 'free'
    face_bc = ['free', 'free', 'free']

    # Specify whether DREAM.3D was previously executed on these files
    # If set to False, the script will NOT generate new DREAM.3D microstructure(s) and instead process the existing microstructures by reading the .csv and GrainID_#.txt files
    # Reasons to set this to False:
    #     1) Process the same set of microstructures with a different number of elements per sub-band, and store these in a separate folder. In this case, copy over the .csv and GrainID_#.txt files to a new folder and run this script.
    #     2) Generate a set of ['periodic', 'periodic', 'periodic'] microstructures, and then reprocesses them with one set of faces set to non-periodic, i.e., ['periodic', 'free', 'periodic'], to study bulk vs. surface fatigue effects. 
    # NOTE: If the variable is set to False, all that is needed in the folder is the DREAM.3D exported .csv and "grainID.txt" files for each microstructure
    generate_new_microstructure_files = True

    # Print the parameters of this microstructure set to a text file
    print_params(directory, size, shape, face_bc, num_vox, band_thickness, num_planes, num_instantiations, d3d_input_file)
    
    # Call to the main function
    gen_microstructures(directory, size, shape, face_bc, num_vox, band_thickness, num_planes, create_sub_bands, num_instantiations, generate_new_microstructure_files, d3d_input_file, d3d_pipeline_path, d3d_executable_path)



if __name__ == "__main__":
    main()
   
# Stopka, K.S., McDowell, D.L. Microstructure-Sensitive Computational Estimates of Driving Forces for Surface Versus Subsurface Fatigue Crack Formation in Duplex Ti-6Al-4V and Al 7075-T6. JOM 72, 28–38 (2020). https://doi.org/10.1007/s11837-019-03804-1

# Stopka and McDowell, “Microstructure-Sensitive Computational Multiaxial Fatigue of Al 7075-T6 and Duplex Ti-6Al-4V,” International Journal of Fatigue, 133 (2020) 105460.  https://doi.org/10.1016/j.ijfatigue.2019.105460

# Stopka, K.S., Gu, T., McDowell, D.L. Effects of algorithmic simulation parameters on the prediction of extreme value fatigue indicator parameters in duplex Ti-6Al-4V. International Journal of Fatigue, 141 (2020) 105865.  https://doi.org/10.1016/j.ijfatigue.2020.105865

# Castelluccio, G.M., McDowell, D.L. Assessment of small fatigue crack growth driving forces in single crystals with and without slip bands. Int J Fract 176, 49–64 (2012). https://doi.org/10.1007/s10704-012-9726-y
