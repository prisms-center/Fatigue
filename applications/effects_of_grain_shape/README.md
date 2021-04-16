# Effects of grain shape

  These scripts are used to reproduce the analysis and plots for the manuscript entitled <B> Crystal plasticity finite element modeling of grain shape and size effects </B>. This work incorporates grain size and grain shape into a crystal plasticity constitutive model using the PRISMS-Plasticity open-source solver.
  
  Users should first download the data set associated with this manuscript from Materials Commons at https://doi.org/10.13011/m3-f90v-gs55. Three folders entitled "Calibration", "Fatigue", and "Hall-Petch" each contain a "ResultsFiles" and a "SimulationFiles" folder. The two scripts available here should be placed in the "ResultsFiles" directory in the "Fatigue" directory, i.e., "Fatigue\ResultsFiles". This folder contains four folders entitled "NoGrainSize", "WithGrainSize", "plasticity_ellipsoid_Microstructures", and "plots". These two scripts will directly read data from these four folders.
  
  Users can then execute these scripts by opening a command prompt window, navigating to the directory with these folders and scripts, and executing the following commands
 
  ```bash
  python grain_size_effects_compile_FIPs.py
  ```
  or 
  ```bash
  python grain_size_effects_max_PSSR.py
  ```
  
  Users can also execute these scripts in an interactive version of Python. First, execute the "ipython" command in a command prompt window to start an interactive session:
  
  ```bash
  ipython
  ```
  
  Then execute the following commands
  
  ```python
  import grain_size_effects_compile_FIPs as analysis_1 # Import the script
  analysis_1.main() # Execute the 'main()' function
  ```
  or
  ```python
  import grain_size_effects_max_PSSR as analysis_2 # Import the script
  analysis_2.main() # Execute the 'main()' function
  ```  
  
  Alternatively, users can import specific functions from the Python scripts, or copy-and-paste certain functions into the interactive Python session.
  
    
  <B>Reference</B>:  A. Lakshmanan, M. Yaghoobi, K. S. Stopka, and V. Sundararaghavan. Crystal plasticity finite element modeling of grain shape and size effects. (in review).