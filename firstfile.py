import numpy as np
import cv2
from scipy.fftpack import dct, idct
import scipy.stats
import tqdm
import matplotlib.pyplot as plt



def dct2(block,cos_t):
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
    return np.average(image,2,weights=[0.2126,0.7152,0.0722]).astype("float32")

def binomTail(n,k,p):
    sum = 0
    test = 1-scipy.stats.binom.cdf(k-1,n,p)
    return test

def gridDetection(I):
    X,Y,C = np.shape(I)
    L = getLuminance(I)
    votes = computeVotes(L)
    histo = np.histogram(votes,range(-1,65))
    bestGrid = np.argmax(histo[0][1:])
    bestValue = np.max(histo[0])

    NFA = 64*X*Y*np.sqrt(X*Y)*binomTail(int(X*Y/64),int(bestValue/64),1/64.)
    if NFA<1 :
        return bestGrid,NFA,votes
    else :
        return -1,NFA,votes

def regionGrowing(votes,seed,W):
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




def BoundingBox(R):
    R = np.transpose(R)
    # print("R",R)
    xmin = np.min(R[0,:])
    xmax = np.max(R[0,:])
    ymin = np.min(R[1,:])
    ymax = np.max(R[1,:])

    # print("MAX",xmin,xmax,ymin,ymax)

    return xmin,xmax,ymin,ymax


def forgeryDetection(votes,G,W):
    forgerMask=np.zeros(np.shape(votes))
    X,Y = np.shape(votes)

    for x in tqdm.tqdm(range(len(forgerMask))):
        for y in range(len(forgerMask[0])):
            if votes[x,y]>-1 and votes[x,y]!=G:
                R = regionGrowing(votes,[x,y],W)
                if len(R)<4:
                    continue
                xmin,xmax,ymin,ymax = BoundingBox(R)
                Bx =  (xmax-xmin)+1
                By = ymax-ymin+1
                N = max(xmax-xmin+1,ymax-ymin+1)
                card = len(R)
                #NFA = 64 *X*Y*np.sqrt(X*Y)*binomTail(int(N*N/64),int(card/64),1/64) formule fausse du papier
                NFA = 64 *Bx*By*np.sqrt(Bx*By)*binomTail(int(N*N/64),int(card/64),1/64.0)
                if NFA < 1 :
                    # forgerMask[np.transpose(R)]=True
                    for i,j in R:
                        forgerMask[i,j]=True
                for i,j in R :
                    votes[i,j] = -1

                    # forgerMask[np.transpose(R)]=True
    return forgerMask



if __name__=="__main__":
  image=cv2.imread("./im2gimp.jpg", cv2.IMREAD_UNCHANGED)
  image=image[:,4:]
  G,value,votes = gridDetection(image)
  if G==-1 :
      print("ERREUR")
  else :
      print(f"Meilleur vote {G}")
      mask = forgeryDetection(votes,G,12)
      # print(votes)
      plt.figure(2)
      plt.imshow(mask)
      plt.figure(0)
      plt.imshow(votes)
      plt.figure(1)
      plt.imshow(votes==-1)
      plt.show()
