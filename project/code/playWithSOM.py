import torch

import visdom

from som import *
from som_utils import *

import time


# Visualization parameters
VIS = visdom.Visdom()
ENV = 'SOM'


# Initial SOM parameters
ROWS = 8
COLS = 8
LR = 0.2
SIGMA = 0.4
SHAPE = 'sphere'
N = 500


def generateSphereSurface(N):
	data = torch.randn(N, 3)
	data /= torch.norm(data, 2, 1, True)
	return data

def generateCubeVolume(N):
	return 2 * torch.rand(N, 3) - 1

def generateSquareArea(N):
	return 2 * torch.rand(N, 2) - 1

def generateCirclePerimeter(N):
	data = 2 * torch.randn(N, 2) - 1
	data /= torch.norm(data, 2, 1, True)
	return data


def parse_shape(shape):
	# Parse test data distribution
	if shape == 'circle':
		dim = 2
		init_shape = '2dgrid'
	elif shape == 'sphere':
		dim = 3
		init_shape = 'ball'
	elif shape == 'cube_vol':
		dim = 3
		init_shape = 'ball'
	elif shape == 'square':
		dim = 2
		init_shape = '2dgrid'

	return dim, init_shape


def iterative_main():

	# Parse test data distribution
	dim, init_shape = parse_shape(SHAPE)

	# Create SOM
	lr = LR
	sigma = SIGMA
	vis = Viz(VIS, ENV)
	som = BatchSOM(ROWS, COLS, dim, vis)
	som.initialize(init_shape)

	# Store the initial SOM contents for visualization purposes
	init_contents = som.contents.clone()

	for i in range(10000):
		# Generate some test data
		if SHAPE == 'circle':
			data = generateCirclePerimeter(N)
		elif SHAPE == 'sphere':
			data = generateSphereSurface(N)
		elif SHAPE == 'cube_vol':
			data = generateCubeVolume(N)
		elif SHAPE == 'square':
			data = generateSquareArea(N)

		# Put data on GPU
		data = data.cuda()

		# Update the SOM
		res = som.update(data, sigma, True)

		# Decay the parameters
		if i % 500 == 0:
			lr *= 0.9
			sigma *= 0.9
			print('New LR: ', lr)
			print('New Sigma: ', sigma)

		# Visualize the SOM
		if i % 5 == 0:
			som.update_viz(
				init_contents.cpu(),
				som.contents.cpu(),
				data.cpu())
			print('Res: ', res)



def parbatch_main():
	if torch.cuda.is_available():
		print("Using CUDA")
		device = torch.device("cuda:0")
	else:
		device = torch.device("cpu")

	# Parse test data distribution
	dim, init_shape = parse_shape(SHAPE)

	# Create SOM
	lr = LR
	sigma = SIGMA
	batches = 4
	# vis = Viz(VIS, ENV)
	vis = None
	som = ParallelBatchSOM(ROWS, COLS, dim, batches, vis)
	som.initialize(init_shape)

	# Store the initial SOM contents for visualization purposes
	init_contents = som.contents.clone()

	start = time.time()
	for i in range(5000):
		# Generate some test data
		if SHAPE == 'circle':
			data = generateCirclePerimeter(N)
		elif SHAPE == 'sphere':
			data = generateSphereSurface(N)
		elif SHAPE == 'cube_vol':
			data = generateCubeVolume(N)
		elif SHAPE == 'square':
			data = generateSquareArea(N)

		# Put data on GPU
		data = data.to(device)
		data = data.repeat(batches, 1, 1)

		# Update the SOM
		res = som.update(data, sigma, True)
		# if i%100==0:
		# 	som.update_viz(
		# 		init_contents.cpu(),
		# 		som.contents.cpu(),
		# 		data.cpu()
		# 	)


	print('Time:', time.time() - start)
	print('Res:', res)
	# som.update_viz(
	# 	init_contents.cpu(),
	# 	som.contents.cpu(),
	# 	data.cpu())




if __name__ == '__main__':
	parbatch_main()
