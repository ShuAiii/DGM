# Deep Galerkin Method

Deep Galerkin method is a deep learning framework for solving partial 
differential equations. This particular implementation can be run, straight 
out-of-the-box in Docker containers. It is also highly customizable.

## Install
`gh repo clone ShuAiii/DGM`

## Build and run Docker container
`make build_dgm` will pull the base tensorflow image and build the DGM source
over the base image. `make dgm` will run a container of the image with gpu capability, 
`make dgm_cpu` will run a cpu only container.

## Example
### Black-Scholes call option
Inside the container, run:
<br />
`python train.py`




