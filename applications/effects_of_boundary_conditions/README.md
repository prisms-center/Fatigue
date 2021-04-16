# Effects of boundary conditions

  These scripts are used to reproduce the analysis and plots for the manuscript entitled <B> Effects of boundary conditions on microstructure-sensitive fatigue crystal plasticity analysis</B>. In this work, the effects of various periodic and non-periodic boundary conditions are evaluated using the macroscopic stress-strain response, measures of local slip, and the extreme value fatigue response. 
  
  Users should first download the data set associated with this manuscript from Materials Commons at https://doi.org/10.13011/m3-mhgc-ec71. The data set consists of four folders entitled "paper_#2_section_3", "paper_#2_section_4", "paper_#2_section_5", and "simulation_files_only". These two scripts should be placed in the same directory as these four folders. Users can then execute these scripts by opening a command prompt window, navigating to the directory with these folders and scripts, and executing the following commands:
 
  ```bash
  python PRISMS_effects_of_BCs_compile_FIPs.py
  ```
  or 
  ```bash
  python PRISMS_effects_of_BCs_local_and_stress_strain.py
  ```
  
  Users can also execute these scripts in an interactive version of Python. First, execute the "ipython" command in a command prompt window to start an interactive session:
  
  ```bash
  ipython
  ```
  
  Then execute the following commands:
  
  ```python
  import PRISMS_effects_of_BCs_compile_FIPs as analysis_1 # Import the script
  analysis_1.main() # Execute the 'main()' function
  ```
  or
  ```python
  import PRISMS_effects_of_BCs_local_and_stress_strain as analysis_2 # Import the script
  analysis_2.main() # Execute the 'main()' function
  ```
  
  Alternatively, users can import specific functions from the Python scripts, or copy-and-paste certain functions into the interactive Python session.
  
  
  <B>Reference</B>:  K. S. Stopka, M. Yaghoobi, J. E. Allison, and D. L. McDowell. Effects of boundary conditions on microstructure-sensitive fatigue crystal plasticity analysis. (in review).