# stitch-aerial-photos
An algorithm that handles large-scale aerial photo co-registration, based on SURF, RANSAC and PyTorch autograd. Extended base repository to accomodate RGB images, add visualization utilities and use to metadata from images to initialize image positions. 

# Data 
The data is available at this [link](https://www.dropbox.com/sh/406k63ojc4jtqpz/AADKE3QsmXb5TEo1e2KlgGQ_a?dl=0). Extract zip. For each subfolder corresponding to a specific parking lot, we have access to images and associated metadata from surveys at several different time intervals. 

# Use 
To stitch a set of images, add folder of images to the data folder. Change the string in the demo notebook to point to directory. Run demo notebook.



