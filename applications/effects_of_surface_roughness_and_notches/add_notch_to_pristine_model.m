%%
%% Written by Mohammadreza Yaghoobi and Krzysztof S. Stopka 
% Initialize 
clc;
clear;

% Specify grain ID input file that should be modified
grainID = load("./grainID_0_5120_grains_random.txt");

%%
% Specify number of voxels in the X, Y, and Z directions
% NOTE: This and the line below should match the dimensions of the .txt
% file specified above! Alternatively, users can edit this script to read in
% these values directly from the script above...
X_number = 128; Y_number = 160; Z_number = 128;

% Specify the physical size of the microstructure model
% As a note, you as the user should keep track of these sizes!
Size_X = 0.4; Size_Y = 0.5; Size_Z = 0.4;

% Get step size in the X, Y, and Z directions
div_X = Size_X/X_number; div_Y = Size_Y/Y_number; div_Z = Size_Z/Z_number;

%%
% Initialize matrix of grain IDs at each element
grainIDMat = zeros(X_number,Y_number,Z_number);
Coordinate = zeros(X_number,Y_number,Z_number,3);

% A different notch radius or depth can be specified as shown below:

%%%%%%%%% Hole same depth same radius(half a circle with the original radius)
hole_radius = 0.3;
center_X = 0;
%%%%%%%%%%Hole half depth same radius
% hole_radius = 0.15+0.225;
% center_X = -0.225;
%%%%%%%%%%Hole one-fourth depth same radius
% hole_radius = 0.15/2+0.5625;
% center_X = -0.5625;
%%%%%%%%%%Hole same depth half radius
% hole_radius = 0.15;
% center_X = 0;
%%%%%%%%%%Hole half depth half radius
% hole_radius = 0.15/2+0.1125;
% center_X = -0.1125;
%%%%%%%%%%Hole one-fourth depth half radius
% hole_radius = 0.15/4+0.28125;
% center_X = -0.28125;
%%%%%%%%%%Hole one-fourth depth Twice radius
% hole_radius = 0.15+1.125;
% center_X = -1.125;

%%
% Compute the radius of the notch
hole_radius = hole_radius*0.025/0.3;

% Compute the centroid of the notch in the X direction
center_X = center_X*0.025/0.3;

% Compute the centroid of the notch in the Y direction
% I.e., specify that the notch should be in the middle of the Y face
center_Y = Size_Y/2;

% Iterate through each element in the X, Y, and Z
for i = 1:X_number
    for j = 1:Y_number
        for k = 1:Z_number
            
            % Get centroid of all elements
            grainIDMat(i,j,k) = grainID((i-1)*Y_number+j,k);
            Coordinate(i,j,k,1:3) = [div_X/2+(i-1)*div_X div_Y/2+(j-1)*div_Y div_Z/2+(k-1)*div_Z];
            
            % Find all elements whose centroids are within the volume
            % specified by the notch and overwrite these to zero
            if (((Coordinate(i,j,k,1)-center_X)^2+(Coordinate(i,j,k,2)-center_Y)^2)<hole_radius^2)
                grainIDMat(i,j,k) = 0;
            end                
        end
    end
end

% Create new array for the grain IDs at each element
grainID_New=zeros(X_number*Y_number,Z_number);

% Iterate through each element in the X, Y, and Z
for i = 1:X_number
    for j = 1:Y_number
        for k = 1:Z_number
            
            % Write new matrix for the grain ID at each element
            NumXY = (i-1)*(Y_number)+j;
            grainID_New(NumXY,k) = grainIDMat(i,j,k);
        end
    end
end

% Write a new grainID.txt microstructure file for PRISMS-Plasticity
writematrix(grainID_New,'grainID_0_5120_grains_random_add_notch.txt','Delimiter','space');