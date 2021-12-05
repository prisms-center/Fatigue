# Effects of sample size and grain neighborhood

  These scripts are used to reproduce the analysis and plots for the manuscript entitled [Simulated effects of sample size and grain neighborhood on the modeling of extreme value fatigue response](https://doi.org/10.1016/j.actamat.2021.117524). In this work, the convergence of Fatigue Indicator Parameters as a function of sample size is investigated. Structural correlations between the highest FIPs and the grains that manifest these FIPs are then examined. Afterwards, the influence of the grain neighborhood on the extreme value fatigue response at hot-spot grains is studied.
  
  Users should first download the data set associated with this manuscript from Materials Commons at https://doi.org/10.13011/m3-31wm-h036. The data set consists of four folders entitled "Section_3", "Section_4", "Section_5", and "plots". These eight scripts should be placed in the same directory as these four folders. Users can then execute these scripts by opening a command prompt window, navigating to the directory with these folders and scripts, and executing the following command for each script:
 
  ```bash
  python PRISMS_FIP_convergence_1st_highest_FIP_analysis.py
  ```
  
  Users can also execute these scripts in an interactive version of Python. First, execute the "ipython" command in a command prompt window to start an interactive session:
  
  ```bash
  ipython
  ```
  
  Then execute the following commands:
  
  ```python
  import PRISMS_FIP_convergence_1st_highest_FIP_analysis as analyze_FIPs # Import the script
  analyze_FIPs.main() # Execute the 'main()' function
  ```
  
  Alternatively, users can import specific functions from the Python scripts, or copy-and-paste certain functions into the interactive Python session.
  
  
  <B>Reference</B>:  K. S. Stopka, M. Yaghoobi, J. E. Allison, and D. L. McDowell. [Simulated effects of sample size and grain neighborhood on the modeling of extreme value fatigue response](https://doi.org/10.1016/j.actamat.2021.117524). <i>Acta Mater.</i>, <b>224</b>, 117524 (2022).