import urllib
import os
import sys
url='https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda_9.1.85_387.26_linux'
filename = "tools-bin/cuda_9.1.85_387.26_linux.run"
urllib.urlretrieve (url, filename, reporthook=dlProgress)