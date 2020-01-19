import numpy as np
import cv2
from scipy.fftpack import dct, idct
import scipy.stats
import tqdm
import matplotlib.pyplot as plt



def dct2(block,cos_t):
  '''
  Computes the dct2 of a 2D array

  Parameters:
  block (np.array of floats of shape (w,h)):

  Returns:
  np.array of float with shape (w,h): 2D dct2 of the block parameter
  '''
  dct_num=np.zeros(block.shape)
  dct_test = dct(dct(block.T, norm='ortho').T, norm='ortho')
  # for i in range(8): #### Sanity check
  #   for j in range(8):
  #     if(i+j>0):
  #       for xx in range(8):
  #         for yy in range(8):
  #           dct_num[i,j]+= block[xx,yy]*cos_t[xx][i]*cos_t[yy][j]
  #       if(i==0):
  #         dct_num[i,j]/= np.sqrt(2)
  #       if(j==0):
  #         dct_num[i,j]*= 1/np.sqrt(2)
  #       dct_num[i,j]*= 0.25
  #       print(dct_num[i,j]-dct_test[i,j])
  return dct_test

def computeVotes(L):
  '''
  Computes for each pixel the best 8x8 jpeg grid

  Parameters:
  L (np.array of floats with shape(w,h): luminance of an image)

  Returns:
  np.array of int with shape(w,h): Best 8x8 grid for each pixel, if best grid is (i,j) the value in the array is i%8 * 8 + j%8
  '''
  votes=np.full(L.shape,-1)
  zeros=np.zeros(L.shape)
  cos_t=np.zeros((8,8))
  for i in range(8):
    for j in range(8):
      cos_t[i,j]=np.cos((2*i+1.0)*j*np.pi/16)
  for i in tqdm.tqdm(range(L.shape[0]-7)):
    for j in range(L.shape[1]-7):
      d=dct2(L[i:i+8,j:j+8],cos_t)
      z=np.sum(abs(d)<0.5)
      for x in range(i,i+8):
        for y in range(j,j+8):
          if z==zeros[x,y]:
            votes[x,y]=-1
          elif z>zeros[x,y]:
            zeros[x,y]=z
            votes[x,y]=i%8 * 8 + j%8
  return votes

def getLuminance(I):
  '''
  Returns the luminance of image I with weights [0.299,0.587,0.114]

  Parameters:
  I (array of shape(w,h,3)): Image with RGB channels

  Returns:
  np.array of shape (w,h): Luminance of I 
  '''
  return np.average(image,2,weights=[0.299,0.587,0.114]).astype("float32")

def binomTail(n,k,p):
  '''
  Returns the tail of order k of a binomial distribution B(n,p)

  Parameters: 
  n (int) 
  k (int) 
  p (float) : probability 0<=p<=1
  
  Returns:
  float: tail of order k of a binomial distribution B(n,p)
  '''
  return 1-scipy.stats.binom.cdf(k-1,n,p)

def gridDetection(I):
  '''
  Returns the best 8x8 jpeg grid for image I

  Parameters:
  I (image of shape(w,h,3)): Image to study

  Returns:
  bestGrid (int): index of the best grid for that image -1 if no significant grid is detected
  NFA (float): value of the NFA for that grid
  votes (array of int with shape (w,h)): best grid for each pixel
  '''
  X=np.shape(I)[0]
  Y=np.shape(I)[1]
  if(len(np.shape(I))==3):
    L = getLuminance(I)
  else:
    L=I
  votes = computeVotes(L)
  histo = np.histogram(votes,range(-1,65))
  bestGrid = np.argmax(histo[0][1:]) ##ignore votes=-1 on the histogram
  bestValue = np.max(histo[0][1:])
  NFA = 64*X*Y*np.sqrt(X*Y)*binomTail(int(X*Y/64),int(bestValue/64.),1/64.)
  print("best grid NFA", NFA)
  if NFA<1 :
      return bestGrid,NFA,votes
  else :
      return -1,NFA,votes

def regionGrowing(votes,seed,W):
  """
  Computes the region of pixels with same vote as the seed (we look for pixel in a window (W,W) to grow the region

  Parameters:
  votes (2D-array of int with shape (w,h)): best grid for each pixel
  seed (tuple of int): pixel to grow the region around
  W (int): size of the window to grow the region

  Returns:
  array of int: list of pixels belonging to the region 
  """
  queue = [seed]
  visited = np.zeros(votes.shape)
  region= [seed]
  while len(queue)>0 :
      x,y=queue.pop(0)
      visited[x,y]=True
      for i in range(-W,W+1):
          for j in range(-W,W+1):
              if (i==0 and j==0) or x+i<0 or x+i>=np.shape(votes)[0] \
              or y+j<0 or y+j>= np.shape(votes)[1]:
                  continue
              elif not visited[x+i,y+j] and votes[x+i,y+j]==votes[x,y] :
                  visited[x+i,y+j]=True
                  region.append([x+i,y+j])
                  queue.append([x+i,y+j])
  return region




def boundingBox(R):
  '''
  Computes the bounding box of a list of pixel indexes

  Parameters:
  R (list of tuple of int)

  Returns:
  (int,int,int,int): bounding box of the region
  '''
  R = np.transpose(R)
  xmin = np.min(R[0,:])
  xmax = np.max(R[0,:])
  ymin = np.min(R[1,:])
  ymax = np.max(R[1,:])

  return xmin,xmax,ymin,ymax


def forgeryDetection(votes,G,W):
  '''
  Computes the forgery zones

  Parameters:
  votes (2D array of int): best grid for each pixel
  G (int): main grid of the image 
  W (int): size of the window for the region growing

  Returns:
  2D array of bool: Binary image expressing whether each pixel has been detected as forged
  '''
  forgerMask=np.zeros(np.shape(votes))
  X,Y = np.shape(votes)

  for x in tqdm.tqdm(range(len(forgerMask))):
      for y in range(len(forgerMask[0])):
          if votes[x,y]>-1 and votes[x,y]!=G:
              R = regionGrowing(votes,[x,y],W)
              if len(R)<4:
                  continue
              xmin,xmax,ymin,ymax = boundingBox(R)
              Bx =  (xmax-xmin)+1
              By = ymax-ymin+1
              N = max(xmax-xmin+1,ymax-ymin+1)
              card = len(R)
              #NFA = 64 *X*Y*np.sqrt(X*Y)*binomTail(int(N*N/64),int(card/64),1/64) formule fausse du papier
              NFA = 64 *Bx*By*np.sqrt(Bx*By)*binomTail(int(N*N/64),int(card/64),1/64.0)
              if NFA < 1 :
                  forgerMask[tuple(np.transpose(R))]=True
              votes[tuple(np.transpose(R))]=-1
  return forgerMask

def getColorDict():
  dic={}
  for i in range(-1,65,1):
    dic[i]=tuple(np.random.random(size=3)*255)
  return dic

def voteColorMap(votes):
  dic=getColorDict()
  colorMapper = lambda value: dic[value]
  return np.moveaxis(np.vectorize(colorMapper)(votes),0,-1).astype('int')

if __name__=="__main__":
  image=cv2.imread("./images/pelican_tampered.ppm", cv2.IMREAD_UNCHANGED)
  plt.figure(0)
  plt.imshow(image,cmap="gray")
  G,value,votes = gridDetection(image)
  plt.figure(1)
  plt.imshow(voteColorMap(votes))
  if G==-1 :
      print("No grid detected.")
  else :
      print(f"Meilleur vote {G}")
  mask = forgeryDetection(votes,G,12)
  plt.figure(2)
  plt.imshow(mask,cmap="gray")


  plt.show()
