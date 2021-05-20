# Generate simple microstructure

  This short script and associated DREAM.3D .json file will generate a simple microstructure with a 3D array of grains with some number of elements per grain. The current setting in the script will create a microstructure with 4 x 10 x 7 grains with 27 elements per grain. This microstructure is shown below.
  
  To execute the script, users should navigate to the directory that contains this script and execute the following command using command prompt.
 
  ```bash
  python generate_simple_ms.py
  ```
  
  Users can also execute these scripts in an interactive version of Python. First, execute the "ipython" command in a command prompt window to start an interactive session:
  
  ```bash
  ipython
  ```
  
  Then execute the following commands:
  
  ```python
  import generate_simple_ms as gen_ms # Import the script
  gen_ms.main() # Execute the 'main()' function
  ```
  
  Example of microstructure

  <B>Reference</B>:  G. M. Castelluccio and D. L. McDowell, "Microstructure and mesh sensitivities of mesoscale surrogate driving force measures for transgranular fatigue cracks in polycrystals," Materials Science and Engineering: A, 639, 626 (2015)