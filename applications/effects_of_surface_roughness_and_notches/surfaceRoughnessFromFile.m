%%
%% Written by Mohammadreza Yaghoobi and Krzysztof S. Stopka 
% Initialize 
clc;
clear;

% This script will read in an existing surface roughness profile and scale
% it by some factor.

%%
% Load in .mat file with an existing surface roughness profile
load('Surface_Roghnsess_half.mat');

% Specify the factor for the new, desired surface roughness profile
% e.g., Factor = 2 --> this will double the height of peaks and depths of
% valleys, whereas Factor = 1/2 will create a roughness profile that is
% half as intense.
Factor = 2; 

%%
F=Factor*F;


% Specify grain ID input file that should be modified
grainID = load("./grainID_0.txt");

%%
% Specify number of voxels in the X, Y, and Z directions
% NOTE: This and the line below should match the dimensions of the .txt
% file specified above! Alternative, users can edit this script to read in
% these values directly from the script above...
X_number = 128; Y_number = 160; Z_number = 128;

% Specify the physical size of the microstructure model
% As a note, you as the user should keep track of these sizes!
Size_X = 0.4; Size_Y = 0.5; Size_Z = 0.4;

% Get step size in the X, Y, and Z directions
div_X = Size_X/X_number; div_Y = Size_Y/Y_number; div_Z = Size_Z/Z_number;


%%
% Specify number of elements that should be added in the direction
% perpendicular to the rough surface
NumberAddedXdir = ceil(max(max(F))/div_X)+1;

% Initialize matrix of grain IDs at each element
grainIDMat = zeros(X_number,Y_number,Z_number);
Coordinate = zeros(X_number,Y_number,Z_number,3);


% Iterate through each element in the X, Y, and Z
for i = 1:X_number
    for j = 1:Y_number
        for k = 1:Z_number
            
            % Get centroid of all elements
            grainIDMat(i,j,k) = grainID((i-1)*Y_number+j,k);
            Coordinate(i,j,k,1:3) = [div_X/2+(i-1)*div_X div_Y/2+(j-1)*div_Y div_Z/2+(k-1)*div_Z];       
        end
    end
end

% Create new array for the grain IDs at each element
grainIDMat_WithBump = zeros(X_number+NumberAddedXdir,Y_number,Z_number);
Coordinate_WithBump = zeros(X_number+NumberAddedXdir,Y_number,Z_number,3);
grainIDMat_WithBump(NumberAddedXdir+1:end,:,:) = grainIDMat(:,:,:);
Coordinate_WithBump(NumberAddedXdir+1:end,:,:,:) = Coordinate(:,:,:,:);

for i=1:NumberAddedXdir
    grainIDMat_WithBump(i,:,:)=grainIDMat(1,:,:);
    Coordinate_WithBump(i,:,:,2)=Coordinate(1,:,:,2);
    Coordinate_WithBump(i,:,:,3)=Coordinate(1,:,:,3);
    Coordinate_WithBump(i,:,:,1)=Coordinate(1,:,:,1)-(NumberAddedXdir-i+1)*div_X;
end

for j=1:Y_number
    for k=1:Z_number
        NumberCells=round(F(k,j)/div_X);
        NumbeZero=NumberAddedXdir-NumberCells;
        grainIDMat_WithBump(1:NumbeZero,j,k)=0;
    end
end


grainID_New=zeros((X_number+NumberAddedXdir)*Y_number,Z_number);
for i=1:X_number+NumberAddedXdir
    for j=1:Y_number
        for k=1:Z_number
            NumXY=(i-1)*(Y_number)+j;
            grainID_New(NumXY,k)=grainIDMat_WithBump(i,j,k);
        end
    end
end

 
% Write a new grainID.txt microstructure file for PRISMS-Plasticity
writematrix(grainID_New,'grainID_With_RandomSurface_half_new.txt','Delimiter','space');

% Save the generated surface to a .mat file
save('Surface_Roughness_half_new','Y','Z','F','x','y','f');
