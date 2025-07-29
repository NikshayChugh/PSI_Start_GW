# PSI_Start_GW
This repository is a collection of code **Taissia Karasova** and me used during summer 2025 to:
1. Compute the Power Spectrum of Galaxies, Dark Matter, and other objects if their field is specified.
2. Generate Mock Siren Catalogs on the Illustris TNG300-1 simulation data, using a broken power law.
3. Use these catalogs to compute the gravitational wave bias for our model.
4. Analyse the obtained data and draw conclusions.

Most of the code in the siren catalog generation is due to **Dorsa Sadat Hossein**. 

This project was supervised by **Prof Ghazal Geshnizjani** at the **Perimeter Institute for Theoretical Physics**, Waterloo, Canada. 
We thank the **PSI Start Program** for funding and support.

## [codes](PSI_Start_GW/codes/) 
This directory contains the tools used to work with the Illustris API, and then to run the computation and analysis. 

### [Python Scripts](PSI_Start_GW/codes/Python Scripts/)
Most relevant scripts are found here.

### [SLurm Scripts](PSI_Start_GW/codes/Slurm Scripts/)
These are slurm job scripts run on `symmetry`, the cluster at PI, to efficiently parallely process the Python jobs.

### [Jupyter Notebooks](PSI_Start_GW/codes/Jupyter Notebooks/)
Miscellaneous notebooks in which I brainstorm and develop how the procedure works.

### [Extra](PSI_Start_GW/codes/Extra/)
Some tools and ideas that were not used much but may be relevant for debugging (look into description for details). 

## [plots](PSI_Start_GW/plots/)
Important and self-containted plots summarise the analysis. A dictionary between snapshot number and redshift may be found at the [Illustris Data Access](https://www.tng-project.org/data/) Page under the Illustris TNG 300-1 simulation details. 
These plots deal with mostly b(0.1 h/Mpc) = b_{GW}. 

