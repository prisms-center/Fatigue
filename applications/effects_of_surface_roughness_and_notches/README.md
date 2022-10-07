# Effects of surface roughness and notches

  These scripts are used to reproduce the microstructure models for the manuscript entitled [Microstructure-sensitive modeling of surface roughness and notch effects on extreme value fatigue response](https://github.com/prisms-center/Fatigue/tree/main/applications). In this work, digital microstructure models are subjected to crystal plasticity finite element method (CPFEM) simulations to examine the combined effects of microstructure and grain size scale asperities on the driving force for fatigue crack formation. Following elastic-plastic shakedown, mesoscale volume-averaged fatigue indicator parameters (FIPs) are computed within fatigue damage process zones of grains. An increase in the intensity of realistic surface roughness profiles corresponds accordingly to larger FIPs but it is difficult to precisely interpret the mechanisms and effects of this intensification. We thus parametrically investigate surface asperities (i.e., notches) that couple with microstructure and lack of constraint on slip at the free surface to describe microstructure-sensitive surface roughness knockdown effects on fatigue resistance.
  
## Generate surface roughness profiles to overlay onto existing digital microstructure models

  This section details the use of the matlab scripts provided here to generate realistic surface roughness profiles and how to overlay these onto existing microstructure models that may then be simulated using PRISMS-Plasticity. The user first needs to generate the ```grainID.txt``` file using either the ```generate_microstructures.py``` script of PRISMS-Fatigue or an equivalent method. The entire code used to generate the surface roughness profiles is available at [http://www.mysimlabs.com/](http://www.mysimlabs.com/) but the necessary scripts are included in this repository. Users will modify the  ```add_surface_roughness_to_pristine_model.m``` script by changing the following inputs:
  
   ### Inputs:
 1. ```grainID.txt``` file which specifies the grain ID of each element in the microstructure model that should be modified.
 1. Size and shape (i.e., number of voxels in each direction and the domain size) of the microstructure model specified above.
 1. Root mean square and correlation length for the surface roughness profile that should be overlaid onto the existing microstructure model.
 
 
 ### Outputs:
 1. Modified microstructure model with surface roughness profile.
 
 
 The last part of ```add_surface_roughness_to_pristine_model.m``` will save the generated surface roughness profile to a ```.mat``` file. Users can use the ```surfaceRoughnessFromFile.m``` script to then read in a previously generated surface roughness ```.mat``` file and scale the profile to generate new models. For example, users can generate multiple surface roughness profiles with the same geometry but for which the peaks and valleys are only half as tall and deep, respectively, by setting ``` Factor = 0.5 ``` in this script. There are also scripts that will solely plot the generated surface roughness profiles for visualizations, e.g., ```plotSurfaceRoughness_full.m```.

## Generate notch geometry to overlay onto existing digital microstructure models

 This section details the use of the ```add_notch_to_pristine_model.m``` script to generate a microstructure model with a notch geometry. The inputs for this script include:
 
 ### Inputs:
 1. ```grainID.txt``` file which specifies the grain ID of each element in the microstructure model that should be modified.
 1. Size and shape (i.e., number of voxels in each direction and the domain size) of the microstructure model specified above.
 1. Size, X centroid, and Y centroid of the notch that should be overlaid onto the existing microstructure model.
 
 
 ### Outputs:
 1. Modified microstructure model with notch.
 
 
 Users can visualize the modified microstructure mesh files by performing PRISMS-Plasticity "simulations" and quickly terminating after a single iteration, i.e., to visualize the mesh, proceed with a simulation but terminate it immediately so that the output visualization files are generated.
 
 ## Data set
 
  
  The data set associated with this manuscript is available on Materials Commons at https://doi.org/10.13011/m3-gzaj-8566.
 
  <B>Reference:</B> K. S. Stopka, M. Yaghoobi, J. E. Allison, and D. L. McDowell. [Microstructure-sensitive modeling of surface roughness and notch effects on extreme value fatigue response](https://doi.org/10.1016/j.ijfatigue.2022.107295). <i>Int. J. Fatigue</i>, <b>166</b>, 107295 (2023).