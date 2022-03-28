For running Peter's Normalising Flow branch 

1)Installation Instructions

In lxplus (Requires python 3.9, tensorflow, and pylorentz)
  
mkdir AnyName
cd AnyName
git clone --branch Peter https://github.com/xju2/gan4hep.git
cd gan4hep
pip install -e .
pip3 install pylorentz #If not already installed
cd ..
mkdir nf_work
cd nf_work
ln -s /eos/user/p/pfitzhug/AnyName/gan4hep/gan4hep/nf/train_nf.py

#Add the relevent .output file (mc16d_364100_dimuon_0Jets.output) to the nf_work folder then in nf_work run:

python train_nf.py \
--data dimuon_inclusive mc16d_364100_dimuon_0Jets.output TestNP --max-evts 100000

---------------
***************
---------------

2) Display Guide

5 numbers should be printed every epoch that are in order:

Current Epoch, Current Loss, Current Wasserstein Distance, Best Wasserstein Distance, Best Epoch

Alongside these numbers is:

Length of predicted data set pre-cuts:  9999
Length of truth data set pre-cuts:  9999
Length of predicted data set post-cuts:  9085
Fraction of events lost :  0.09140914091409136

This is due to the selection cuts making sure all eta and phi values are between -pi and pi and displaying what events get lost in the process

---------------
***************
---------------

3)Graphs Guide
#Output graphs will be sent to most recent folder in AnyName/nf_work/TestNP/imgs

#There will be a lot of plots!
  
  #image_at_epoch_xxx is the plot of the 6 original and 3 calculated variables with the x axis showing the original values
  #normalised_image_at_epoch_xxx is the plot of the 6 original variables with the x axis having been scaled by StandardScalar -->Off By Default
  #Graph names that start with variables:... are heat plots for the combination of every variable and therefore make up most of the graphs -->Off By Default
  #The three calculated variables each have 2 plots (one log and one non log) for themselves) (variables named 'dimuon' (invariant mass), 'dimuon_pt', and 'dimuon_pseudo')
  #Heatmap_at_epoch_xxx shows the correlation plot for either generated or true data -->Off By Default
  # Log loss and best wasserstein distance record the respective values for each epoch and plots a graph at the end of the training run (Only produced if a run is complete)
