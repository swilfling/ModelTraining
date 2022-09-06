# DataProcessing

This folder contains scripts for data processing in the NextGenHyb2 repository. 

There are two main scripts here:
 - DataProcessing.py
 - Simulation_Solar_Testbench.py

## DataProcessing.py - by Basak:

This script analyzes the `Data/CaseData.csv` file and plots the data. The plots are stored in the folder `Figures`. 
Added recently: Warm water plot

## Simulation_Solar_Testbench.py - by Sandra:

This is a simulation script that couples Python and Dymola. The script executes a Dymola simulation and receives the data from the simulation. 
The script relies on the utility functions that are in the file `SimulationFunctions.py`. 

In order to get the script to work, the following environment variables should be set: 

`HYBRIDCOSIM_REPO_PATH=<path to the hybrid cosim repository>`
`PYTHONPATH=PYTHONPATH;C:\Program Files\Dymola 2021\Modelica\Library\python_interface\dymola.egg`

The script stores its output in the folder `$HYBRIDCOSIM_REPO_PATH/NextGenHyb2/Data/SimulationResults`.

