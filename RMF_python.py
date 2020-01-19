from PIL import *
import numpy as np
import numpy
import os
import cv2
from matplotlib.pyplot import imshow
import numpy as np
import cv2
from scipy.fftpack import dct, idct
import scipy.stats
import tqdm
import matplotlib.pyplot as plt
import copy





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
  votes=np.full((L.shape[0],L.shape[1]),-1)
  zeros=np.zeros(L.shape)
  zerosV2=np.zeros((L.shape[0],L.shape[1],64))
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
            zerosV2[x,y,(i%8)*8+j%8]= z
  zerosV2 =  zerosV2/np.stack([np.sum(zerosV2[:,:,:],axis=-1) for i in range(64)]).reshape((L.shape[0],L.shape[1],64))
  print(len(np.where(zerosV2)[0]))
  zerosV2 = np.where(zerosV2<0,0,zerosV2)
  return votes,zerosV2

def getLuminance(I):
  '''
  Returns the luminance of image I with weights [0.299,0.587,0.114]

  Parameters:
  I (array of shape(w,h,3)): Image with RGB channels

  Returns:
  np.array of shape (w,h): Luminance of I 
  '''
  return np.average(I,2,weights=[0.299,0.587,0.114]).astype("float32")

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
  votes,zeros = computeVotes(L)
  histo = np.histogram(votes,range(-1,65))
  bestGrid = np.argmax(histo[0][1:]) ##ignore votes=-1 on the histogram
  bestValue = np.max(histo[0][1:])
  NFA = 64*X*Y*np.sqrt(X*Y)*binomTail(int(X*Y/64),int(bestValue/64.),1/64.)
  print("best grid NFA", NFA)
  if NFA<1 :
      return bestGrid,NFA,votes,zeros
  else :
      return -1,NFA,votes,zeros




####==============================================================######
def compute_belief(data_cost, msg_up, msg_down, msg_left, msg_right):
    print("compute_belief")
    
    new_cost = copy.deepcopy(data_cost)
    
    shape = np.shape(new_cost)
    
    new_cost[0:-1,0:,:] += msg_up[1:,:,:]
    
    new_cost[1:shape[0],0:shape[1],0:shape[2]] += msg_down[0:shape[0]-1,0:shape[1],0:shape[2]]   
        
    new_cost[0:shape[0],0:shape[1]-1,0:shape[2]] += msg_left[0:shape[0],1:shape[1],0:shape[2]]   
    
    new_cost[0:shape[0],1:shape[1],0:shape[2]] += msg_right[0:shape[0],0:shape[1]-1,0:shape[2]]
    
    belief = new_cost
    
    return belief
    
    
def compute_energy(data_cost, disparity, lambda_value):
    print("compute_energy")
    index_X = np.stack([np.ones((len(data_cost))) * i for i in range(len(data_cost))]).reshape(-1).astype("int")
    index_Y = np.stack([np.ones((len(data_cost))) * i for i in range(len(data_cost[0]))]).transpose().reshape(-1).astype("int")
    # print(index_X.shape)
    # print(type(index_X[0]))
    # print(type(index_X[1]))
    # print(index_Y.shape)
    # print(disparity.shape)
    # print(disparity.flatten().shape)
    # print(type(disparity.flatten()[0]))
    print(np.where(data_cost<0))
    energy = np.sum(data_cost[index_X,index_Y,disparity.flatten().astype(int)])
    shape = np.shape(disparity)
    # print(energy)
    left_energy = np.sum(np.where(disparity[1:shape[0]-1,1:shape[1]-2] != disparity[1:shape[0]-1,1:shape[1]-2],1,0))
    
    right_energy = np.sum(np.where(disparity[1:shape[0]-1,1:shape[1]-1] != disparity[1:shape[0]-1,2:shape[1]],1,0))
    
    up_energy = np.sum(np.where(disparity[1:shape[0]-2,1:shape[1]-1] != disparity[1:shape[0]-1,1:shape[1]-1],1,0))
    
    down_energy = np.sum(np.where(disparity[2:shape[0],1:shape[1]-1] != disparity[1:shape[0]-1,1:shape[1]-1],1,0))
    
    energy += lambda_value*(left_energy + up_energy + right_energy + down_energy)
    
    
    return energy
    
    
def compute_data_cost(img,num_disp_vals = 64) : 
    print("compute_data_cost")
    data_cost,votes = computeVotes(img)
    return data_cost,votes
    
def compute_MAP_labeling(beliefs) :
    print("MAP")
    print(np.shape(beliefs))
    return beliefs.argmin(axis = 2)
    


def normalize_messages(msg_up, msg_down, msg_left, msg_right):
    print("normalize")
    shape = np.shape(msg_up)
    mean_msg_val = msg_up.mean(axis=2)
    for i in range(np.shape(msg_up)[2]):
        msg_up[0:shape[0],0:shape[1],i] =  msg_up[0:shape[0],0:shape[1],i] - mean_msg_val[0:shape[0],0:shape[1]]
    
    
    shape = np.shape(msg_down)
    mean_msg_val = msg_down.mean(axis=2)
    for i in range(np.shape(msg_down)[2]):
        msg_down[0:shape[0],0:shape[1],i] =  msg_down[0:shape[0],0:shape[1],i] - mean_msg_val[0:shape[0],0:shape[1]]
    
    
    shape = np.shape(msg_left)
    mean_msg_val = msg_left.mean(axis=2)
    for i in range(np.shape(msg_left)[2]):
        msg_left[0:shape[0],0:shape[1],i] =  msg_left[0:shape[0],0:shape[1],i] - mean_msg_val[0:shape[0],0:shape[1]]
    
    
    shape = np.shape(msg_right)
    mean_msg_val = msg_right.mean(axis=2)
    for i in range(np.shape(msg_right)[2]):
        msg_right[0:shape[0],0:shape[1],i] =  msg_right[0:shape[0],0:shape[1],i] - mean_msg_val[0:shape[0],0:shape[1]]
        
    return msg_up, msg_down, msg_left, msg_right
    
def update_messages(msg_up_prev, msg_down_prev, msg_left_prev, msg_right_prev, data_cost, lambda_value):
    print("update")

  #Initialisation des messages précèdents :
    msg_up = msg_up_prev
    msg_down = msg_down_prev
    msg_left = msg_left_prev
    msg_right = msg_right_prev
    
    num_disp_vals = np.shape(msg_up)[2]
  
  # Calcul de la somme des messages précèdents et du coût 'naturelle'
  # en fonction du pixel choisi et du type de message à envoyer, les msg ne sont pas définies partout :

    aux_up = msg_up_prev[0:-2,1:-1,:]+msg_left_prev[1:-1,0:-2,:]+msg_right_prev[1:-1,2:,:]+data_cost[1:-1,1:-1,:]
  
    aux_down = msg_down_prev[2:,1:-1,:]+msg_left_prev[1:-1,:-2,:]+msg_right_prev[1:-1,2:,:]+data_cost[1:-1,1:-1,:]
    
    aux_left = msg_up_prev[0:-2,1:-1,:]+msg_down_prev[2:,1:-1,:]+msg_left_prev[1:-1,0:-2,:]+data_cost[1:-1,1:-1,:]
    
    aux_right = msg_up_prev[0:-2,1:-1,:]+msg_down_prev[2:,1:-1,:]+msg_right_prev[1:-1,2:,:]+data_cost[1:-1,1:-1,:]
    
    min_up = (aux_up+lambda_value).min(axis = 2)
    min_down = (aux_down+lambda_value).min(axis = 2)
    min_right =(aux_right+lambda_value).min(axis = 2)
    min_left = (aux_left+lambda_value).min(axis = 2)
    
    # Utilisation de la formule de la question 2 pour chaque type de message :
    for i in range(num_disp_vals):
        msg_up[1:-1,1:-1,i] = np.where(min_up<aux_up[:,:,i],min_up,aux_up[:,:,i])
        msg_down[1:-1,1:-1,i] = np.where(min_down<aux_down[:,:,i],min_down,aux_down[:,:,i])
        msg_right[1:-1,1:-1,i] = np.where(min_right<aux_right[:,:,i],min_right,aux_right[:,:,i])
        msg_left[1:-1,1:-1,i] = np.where(min_left<aux_left[:,:,i],min_left,aux_left[:,:,i]) # Le minimum sera jamais atteint en i.
    
    return msg_up,msg_down,msg_right,msg_left
 
# For convenience do not compute the messages that are sent from pixels that are on the boundaries of the image. 
# Compute the messages only for the pixels with coordinates (y,x) = ( 2:(height-1) , 2:(width-1) )
## CONCATENATION :

def stereo_belief_propagation(img, lambda_value):
    num_disp_values = 64 #number of disparity values
    # tau             = 15 
    num_iterations  = 1000 #number of iterations
    height, width = np.shape(img)

    #compute the data cost term
    #data_cost: a 3D array of size height x width x num_disp_value; each
    #  element data_cost(y,x,l) is the cost of assigning the label l to pixel 
    #  p = (y,x)
    votes,data_cost = compute_data_cost(img)
    print("COMPUTE DATA COST",np.where(data_cost<0))
    #allocate memory for storing the energy at each iteration
    energy = []
    
    #Initialize the messages at iteration 0 to all zeros
    
    #msg_up : a 3D array of size height x width x num_disp_value; each vector
    #  msg_up(y,x,1:num_disp_values) is the message vector that the pixel 
    #  p = (y,x) will send to the pixel up with coordinates q = (y-1,x)
    msg_up    = np.zeros((height, width, num_disp_values))
    #msg_down : a 3D array of size height x width x num_disp_value; each vector
    #  msg_down(y,x,1:num_disp_values) is the message vector that the pixel 
    #  p = (y,x) will send to the pixel down with coordinates q = (y+1,x)
    msg_down  = np.zeros((height, width, num_disp_values))
    #msg_left : a 3D array of size height x width x num_disp_value; each vector
    #  msg_left(y,x,1:num_disp_values) is the message vector that the pixel 
    #  p = (y,x) will send to the pixel left with coordinates q = (y,x-1)
    msg_left  = np.zeros((height, width, num_disp_values))
    #msg_right : a 3D array of size height x width x num_disp_value; each vector
    #  msg_right(y,x,1:num_disp_values) is the message vector that the pixel 
    #  p = (y,x) will send to the pixel right with coordinates q = (y,x+1)
    msg_right = np.zeros((height, width, num_disp_values))
    
    for iter in range(num_iterations):
        print("iter", iter)
        #update messages
        msg_up, msg_down, msg_left, msg_right = update_messages(msg_up, msg_down, msg_left, msg_right, data_cost, lambda_value)
        print("update_messages",np.where(data_cost<0))
        msg_up, msg_down, msg_left, msg_right = normalize_messages(msg_up, msg_down, msg_left, msg_right)
        
        #compute  beliefs
        #beliefs: a 3D array of size height x width x num_disp_value; each
        #  element beliefs(y,x,l) is the belief of pixel p = (y,x) taking the
        #  label l
        beliefs = compute_belief(data_cost, msg_up, msg_down, msg_left, msg_right )
        print("compute_beliefs",np.where(data_cost<0))
        #compute MAP disparities
        #disparity: a 2D array of size height x width the disparity value of each 
        #  pixel; the disparity values range from 0 till num_disp_value - 1
        disparity = compute_MAP_labeling(beliefs)

        #compute MRF energy   
         
        energy.append(compute_energy(data_cost, disparity, lambda_value))
        print("energy",np.where(data_cost<0))
        print(energy[-1])
        # cv2.imshow("DISPARITY ITER {}".format(iter),disparity)
        disparity = (disparity*4).astype("uint8")
        if iter%10 == 0:
            fig = plt.figure()
            plt.imshow(disparity)
            plt.savefig(f'disparity_iter{iter}.jpg')
            plt.close(fig)

        # plt.show()
        
    disparity = (disparity*4).astype("uint8")
    # imshow("DISPARITY ITER {}".format(iter),disparity)
    return disparity,energy


## MAIN :


if __name__ =="__main__":

    # img_left_path  = os.path.join('tsukuba','imL.png')
    # img_right_path = os.path.join('tsukuba','imR.png')
    # disparity_path =  os.path.join('tsukuba','tsukuba-truedispL.png') 

    # #read ground truth disparity image
    # disparity = cv2.imread(disparity_path,cv2.IMREAD_GRAYSCALE)
    # #read left and righ images
    # img_left  = cv2.imread(img_left_path,cv2.IMREAD_GRAYSCALE)
    # img_right = cv2.imread(img_right_path,cv2.IMREAD_GRAYSCALE)
    # kernel = np.ones((5,5),np.float32)/25
    # img_left = cv2.filter2D(img_left,-1,kernel)
    # img_right = cv2.filter2D(img_right,-1,kernel)

    # img_left = numpy.array(img_left)
    # img_right = numpy.array(img_right)




    #estimate the disparity map with the Max-Product Loopy Belief Propagation
    #Algorithm
    image = cv2.imread("./images/pelican_tampered.ppm", cv2.IMREAD_UNCHANGED)
    image = np.average(image,2,weights=[0.299,0.587,0.114]).astype("float32")
    lambda_value = 10
    disparity_est, energy = stereo_belief_propagation(image,lambda_value)
    print(disparity_est)
    disparity_est = disparity_est.astype(np.uint8)
    # figure(3); clf(3);
    # imagesc(disparity_est); colormap(gray)
    # title('Disparity Image');
    cv2.imshow("FINAL",disparity_est)
    # cv2.imshow("Ground TRUTH",disparity)
    # figure(4); clf(4);
    # plot(1:length(energy),energy)
    # title('Energy per iteration')
    import matplotlib.pyplot as plt

    plt.plot(energy)
    plt.show()