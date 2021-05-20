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
  
  The default setting in the script will create the following microstructure, show here with unique grains. As listed, there are 4, 10, and 7 grains in the X, Y, and Z directions, respectively. 
  
  ![grain_IDs](https://user-images.githubusercontent.com/74416866/118996464-8e8a9800-b94d-11eb-8f4c-eacc0fc8954a.png)
  
  The same microstructure with elements displayed is shown below. 
  
  ![grain_IDs_with_elements](https://user-images.githubusercontent.com/74416866/118996809-cdb8e900-b94d-11eb-8dba-0f5cf5946d81.png)
  
  
  This microstructure is sometimes simulated in manuscripts but may be difficult for users to generate. One example of where such a microstructure has been used is the following reference (see Fig. 2):
  
  <B>Reference</B>:  G. M. Castelluccio and D. L. McDowell, "Microstructure and mesh sensitivities of mesoscale surrogate driving force measures for transgranular fatigue cracks in polycrystals," Materials Science and Engineering: A, 639, 626 (2015)
