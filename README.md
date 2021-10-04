# Deep Galerkin Method

Deep Galerkin method is a deep learning framework for solving partial 
differential equations. This particular implementation can be run, straight 
out-of-the-box in Docker containers. It is also highly customizable.

## Install
`git fetch https://github.com/ShuAiii/DGM.git`

## Build and run Docker container
`make build_dgm` will pull the base tensorflow image and build the DGM source
over the base image. `make dgm` will run the image as a container.

## Example
### Black-Scholes call option
Inside the container, run:
<br />
`python train.py`




