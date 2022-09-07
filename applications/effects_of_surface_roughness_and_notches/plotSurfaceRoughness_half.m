clear
clc

%I want to generate a microsttructure with twice roughness hight of surface_roughness_half
load('Surface_Roughness_half.mat');
Factor=2; 

%I want to generate a microsttructure with half a roughness hight of surface_roughness
% load('Surface_Roghnsess.mat');
% Factor=1/2;


F=Factor*F*1000;
f=Factor*f*1000;
x=x*1000+250;
y=(y*1000+250)*0.8;
mesh(x,y,f); 
axis('equal'); 
xlabel('x','FontWeight','bold'), ylabel('y','FontWeight','bold'), zlabel('')
grid('on'), rotate3d('on');
