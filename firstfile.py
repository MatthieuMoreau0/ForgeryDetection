import numpy as np
import cv2
from scipy import fftpack

def computeVotes(L):
  votes=np.full(L.shape,-1)
  zeros=np.zeros(L.shape)
  for i in range(L.shape[0]-8):
    for j in range(L.shape[1]-8):
      d=fftpack.dct(L[i:i+8,j:j+8])
      z=np.sum(d<0.5)
      for x in range(i,i+8):
        for y in range(j,j+8):
          if z==zeros[x,y]:
            votes[x,y]=-1
          elif z>zeros[x,y]:
            zeros[x,y]=z
            votes[x,y]=x%8 * 8 + y%8
  return votes


  return 0

if __name__=="__main__":
  image=cv2.imread("./74692525_474894606706781_1842248289437614080_n.jpg", cv2.IMREAD_UNCHANGED)
  image=cv2.resize(image,(372,496))
  luminance=np.average(image,2,weights=[0.2126,0.7152,0.0722]).astype("float32")
  votes=computeVotes(luminance)
  print(np.histogram(votes,64))
