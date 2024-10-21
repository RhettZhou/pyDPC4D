# pyDPC4D

By Xuyang (Rhett) Zhou, Max-Planck-Institut für Eisenforschung GmbH, x.zhou@mpie.de

## 1.0 Introduction
__________________________________________________________________________________________________________________________________________________________________

pyDPC4D provides a systematic data processing approach for the atomic-resolution differential phase contrast - four-dimensional scanning transmission electron microscopy (DPC-4DSTEM) data sets. With this code, you can reconstruct the annual dark field, electric field, vector field divergence, (projected) electrostatic potential, and charge density images. 

The current code was developed based on py4DSTEM, which can reconstruct the dark field image and calculate the centre of mass of the transmitted beam for each probe position. Our pyDPC4D codes mainly focus on the quantitative calculation of the electrostatic maps. More details on the reconstruction can be found in the attached references and in our publication as follow. 

Xuyang Zhou, Ali Ahmadian, Baptiste Gault, Colin Ophus, Christian H. Liebscher, Gerhard Dehm, Dierk Raabe, Atomic motifs govern the decoration of grain boundaries by interstitial solutes, submitted 2022.

I publish this code for the following two reasons. First, it documents step by step how we handled the 4DSTEM dataset. Second, this code may be helpful to people who want to perform a quantitative analysis of their collected 4DSTEM dataset. It should be mentioned that I did not invent this method. However, I would like to provide a simple way for quantitative data processing, especially for the charge density map.   

To illustrate the workflow, I use an experimental Fe sigma-5 bicrystalline sample with boron/carbon interstitial solutions at the grain boundary. This process can also be applied to the simulated 4DSTEM data set from muSTEM.

Note: These codes are free to use. I would appreciate if you could site our paper. Please also cite the following references as they are the source on which the current pyDPC4D was built.

Reference:

[1] C. Ophus, Microsc. Microanal. 25, 563 (2019).

[2] B. H. Savitzky et al., Microsc. Microanal., 1 (2021). (https://github.com/py4dstem)

[3] K. Müller et al., Nat. Commun. 5, 5653 (2014).

[4] J.A. Hachtel et al., Adv. Struct. Chem. Imag. 4 (2018)  (https://github.com/hachteja/GetDPC)

[5] L. J. Allen et al., Ultramicroscopy 151, 11 (2015). (https://github.com/HamishGBrown/MuSTEM)


## 2.0 Installation
__________________________________________________________________________________________________________________________________________________________________

-> conda create -n pyDPC4D python==3.10

-> conda activate base

-> conda activate pyDPC4D

-> conda install hyperspy==1.7.3 -c conda-forge

-> pip install py4dstem==0.12.24

-> pip install opencv-python==4.6.0.66 

-> pip install pixstem==0.4.0

-> pip install lxml==4.9.1

-> pip install h5py==3.7.0 (I have to reinstall h5py)

-> pip install jupyter==1.0.0

-> pip install numpy==1.23.4

## 3.0 Usage
____________________________________________________________________________________________________________________________________________________________________
Please refer to the jupyter notebook: “pyDPC4D.ipynb”


## Acknowledgments
____________________________________________________________________________________________________________________________________________________________________

I would like to thank Dr. Colin Ophus for his insightful discussion and help with coding.
