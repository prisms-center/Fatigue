%%
%% Written by Mohammadreza Yaghoobi and Krzysztof S. Stopka 
% Initialize 
clc;
clear;

%%
% N is the number of voxels along the Y dimension of the microstructure
% model and rL is a parameter necessary to generate the surface roughness.
% Please see the rsgeng2D and rsgene2D functions for more details.
N = 160;
rL = 0.5*(1-(1/N));

% Specify root mean square (RMS) value of surface roughness
% As a note, you as the user should keep track of these sizes!
% h = 0.01;
h = 0.005;

% Specify correlation length of surface roughness
% As a note, you as the user should keep track of these sizes!
clx = 0.05;

% Call function to generate 2D gaussian (see function for more details)
[f,x,y] = rsgeng2D(N,rL,h,clx);

%% Scale surface roughness values based on size of mesh of interest
% if min(min(f))<0
%     f=f+abs(min(min(f)));
% else
%     f=f-abs(min(min(f)));
% end
F = f(1:128,1:160);
% if min(min(F))<0
%     F=F+abs(min(min(F)));
% else
%     F=F-abs(min(min(F)));
% end
Z = x(1:128);
Y = y;
% Y=Y+abs(min(min(Y)));
% Z=Z+abs(min(min(Z)));

%% Plot surface roughness profile
mesh(x,y,f); 
axis('equal'); 
xlabel('x','FontWeight','bold'), ylabel('y','FontWeight','bold'), zlabel('')
grid('on'), rotate3d('on');


% Specify grain ID input file that should be modified
grainID = load("./grainID_0.txt");

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

for i = 1:NumberAddedXdir
    grainIDMat_WithBump(i,:,:) = grainIDMat(1,:,:);
    Coordinate_WithBump(i,:,:,2) = Coordinate(1,:,:,2);
    Coordinate_WithBump(i,:,:,3) = Coordinate(1,:,:,3);
    Coordinate_WithBump(i,:,:,1) = Coordinate(1,:,:,1) - (NumberAddedXdir-i+1)*div_X;
end

for j = 1:Y_number
    for k = 1:Z_number
        NumberCells = round(F(k,j)/div_X);
        NumbeZero = NumberAddedXdir - NumberCells;
        grainIDMat_WithBump(1:NumbeZero,j,k) = 0;
    end
end


grainID_New = zeros((X_number+NumberAddedXdir)*Y_number,Z_number);
for i = 1:X_number+NumberAddedXdir
    for j = 1:Y_number
        for k = 1:Z_number
            NumXY = (i-1)*(Y_number)+j;
            grainID_New(NumXY,k) = grainIDMat_WithBump(i,j,k);
        end
    end
end

% Write a new grainID.txt microstructure file for PRISMS-Plasticity
writematrix(grainID_New,'grainID_With_RandomSurface_half.txt','Delimiter','space');

% Save the generated surface to a .mat file that can be access later
save('Surface_Roughness_half_new','Y','Z','F','x','y','f');