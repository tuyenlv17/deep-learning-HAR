import urllib
import os
url=raw_input("cuda_url=")
print url
filename = "tools-bin/cuda_9.1.85_387.26_linux.run"
urllib.urlretrieve (url, filename)