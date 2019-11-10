import numpy as np
import cv2
from scipy.fftpack import dct, idct
import scipy.stats
import tqdm
import matplotlib.pyplot as plt



def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def computeVotes(L):
  votes=np.full(L.shape,-1)
  zeros=np.zeros(L.shape)
  for i in tqdm.tqdm(range(L.shape[0]-8)):
    for j in range(L.shape[1]-8):
      d=dct2(L[i:i+8,j:j+8])

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
    # GET IT OUT
    return np.transpose(np.average(image,2,weights=[0.2126,0.7152,0.0722]).astype("float32"))

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
        for i in range(int(-W/2),int(W/2)+1):
            for j in range(int(-W/2),int(W/2)+1):
                if (i==0 and j==0) or x+i<0 or x+i>=np.shape(votes)[0] \
                or y+j<0 or y+j>= np.shape(votes)[1]:
                    continue
                elif not visited[x,y] and votes[x+i,y+j]==votes[x,y] :
                    region.append([x+i,y+j])
                    visited[x+i,y+j]=True
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
                Bx =  (xmax-xmin)
                By = ymax-ymin
                N = max(xmax-xmin,ymax-ymin)
                card = (xmax-xmin)*(ymax-ymin)
                # NFA = 64 *X*Y*np.sqrt(X*Y)*binomTail(int(N*N/64),int(card/64),1/64)
                NFA = 64 *Bx*By*np.sqrt(Bx*By)*binomTail(int(N*N/64),int(card/64),1/64)
                if NFA < 1 :
                    forgerMask[np.transpose(R)]=True
                #     for i in R:
                #         forgerMask[i]=True
                # for i in R :
                #     votes[i] = -1

                    # forgerMask[np.transpose(R)]=True
                votes[np.transpose(R)]=-1
    return forgerMask



if __name__=="__main__":
  image=cv2.imread("./0.jpg", cv2.IMREAD_UNCHANGED)
  image=image[1:]
  G,value,votes = gridDetection(image)
  if G==-1 :
      print("ERREUR")
  else :
      mask = forgeryDetection(votes,G,12)
      print(mask)
      print(np.where(mask))
      plt.imshow(mask*255)
      plt.show()
