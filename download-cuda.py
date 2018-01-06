import urllib
import os
import sys
def dlProgress(count, blockSize, totalSize):
	percent = int(count*blockSize*100/totalSize)
	sys.stdout.write("\r%2d%%" % percent)
	sys.stdout.flush()
url='https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda_9.1.85_387.26_linux'
filename = "tools-bin/cuda_9.1.85_387.26_linux.run"
urllib.urlretrieve (url, filename, reporthook=dlProgress)