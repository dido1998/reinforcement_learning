#!/usr/bin/env python3

from __future__ import division, print_function

import sys
import numpy as np
import gym
import time
import tensorflow as tf
from optparse import OptionParser
from random import shuffle
import gym_minigrid


def weights(shape):
    initial=tf.truncated_normal(shape,0.0,stddev=1e-1)
    return tf.Variable(initial,trainable=True)
def biases(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def conv2d(w,x):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

data=tf.placeholder(tf.float32,shape=[7,7,3,None])
datav=tf.reshape(data,[-1,7,7,3])
w11=tf.Variable(tf.truncated_normal([7,7,3,4],0.0,stddev=1e-1),name='w11')
#b11=tf.Variable(tf.constant(0.1,shape=[4]),name='b11')
h11=conv2d(w11,datav)
h11_r=tf.nn.relu(h11)

"""w12=tf.Variable(tf.truncated_normal([7,7,4,4],0.0,stddev=1e-1),trainable=True,name='w12')
b12=tf.Variable(tf.constant(0.1,shape=[4]),trainable=True,name='b12')
h12=conv2d(w12,h11_r)+b12
h12_r=tf.nn.relu(h12)
"""





iptornn=tf.reshape(h11_r,[-1,7*7*4])
t_size=tf.shape(iptornn)[0]
iptornn=tf.expand_dims(iptornn,axis=1)
iptornn=tf.transpose(iptornn,perm=[1,0,2])

lstm = tf.nn.rnn_cell.LSTMCell(16) 

outputs, state=tf.nn.dynamic_rnn(cell=lstm,inputs=iptornn,dtype=tf.float32)
outputs=tf.nn.relu(outputs)

outputs=tf.squeeze(outputs,0)
outputs=tf.nn.relu(outputs)
inputv,inputp=tf.split(outputs,2,1)


w31v=tf.Variable(tf.truncated_normal([8,8],0.0,stddev=1e-1),name='w31v')
b31v=tf.Variable(tf.constant(0.1,shape=[8]),name='b31v')
v1=tf.matmul(inputv,w31v)+b31v
v1_r=tf.nn.relu(v1)

w32v=tf.Variable(tf.truncated_normal([8,1],0.0,stddev=1e-1),trainable=True,name='w32v')
b32v=tf.Variable(tf.constant(0.1,shape=[1]),trainable=True,name='b32v')
v=tf.matmul(v1_r,w32v)+b32v
target_v=tf.placeholder(tf.float32,shape=[None,1])
lossv=tf.reduce_sum(tf.square(tf.subtract(target_v,v)))




w31p=tf.Variable(tf.truncated_normal([8,8],0.0,stddev=1e-1),name='w31p')
b31p=tf.Variable(tf.constant(0.1,shape=[8]),name='b31p')
p1=tf.matmul(inputp,w31p)+b31p
p1_r=tf.nn.relu(p1)

w32p=tf.Variable(tf.truncated_normal([8,4],0.0,stddev=1e-1),name='w32p')
b32p=tf.Variable(tf.constant(0.1,shape=[4]),name='b32p')
p=tf.matmul(p1_r,w32p)+b32p

probs=tf.nn.softmax(p)


target_actions=tf.placeholder(tf.float32,shape=[None,4])
entropy=tf.reduce_sum(probs*tf.log(probs))
advantage=tf.placeholder(tf.float32,shape=[None,1])
g_probs=tf.reduce_sum(tf.multiply(probs,target_actions),reduction_indices=[1])
l_probs=tf.log(g_probs)
ent_constant=tf.placeholder(tf.float32)
lossp=-tf.reduce_sum(tf.multiply(advantage,l_probs))+1e-1*entropy

loss=tf.add(lossv,lossp)

optimize=tf.train.AdamOptimizer(1e-4).minimize(loss)
init=tf.global_variables_initializer()

saver=tf.train.Saver()


z1=False
g=False
def modifyreward(s,s1,anow):
    global z1
    r=0
    t=False
    """
    if stp1 is None and stp2 is None:
        stp1=s
        stp2=s1
    else:        if np.all(np.equal(stp1,s1)):
            r-=30
        stp1=s
        stp2=s1"""
    gate_pos_x_s=None
    gate_pos_y_s=None
    gate_pos_x_s1=None
    gate_pos_y_s1=None
    for i in range(7):
        for j in range(7):
            if s[i,j,0]==2 and s[i,j,2]==0:
                gate_pos_x_s=i
                gate_pos_y_s=j
            if s1[i,j,0]==2 and s1[i,j,2]==0:
                gate_pos_x_s1=i
                gate_pos_y_s1=j
    if gate_pos_y_s1 is not None and gate_pos_x_s1 is not None and gate_pos_y_s is not None and gate_pos_x_s is not None:
        if ((3-gate_pos_x_s)**2+(6-gate_pos_y_s))**2>((3-gate_pos_x_s1)**2+(6-gate_pos_y_s1))**2:
            r+=1
        else:
            r=-1

  
    if t==True and anow==2:
        r+=60
            
        t=False
    
    if s1[3,5,0]==2 and s1[3,5,2]==0   :
        r+=50
        
    if s[3,5,0]==2 and s[3,5,2]==0 and anow==3 and z1==False  :
        r+=60
        t=True
        z1=True    

            

    return r



def main():
    global z1
    countsuccess=0
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-MultiRoom-N6-v0'
    )
    (options, args) = parser.parse_args()
    # Load the gym environment
    env = gym.make('MiniGrid-MultiRoom-N6-v0')
    s=env.reset()
    
    # Create a window to render into
    renderer = env.render('human')
    
          

            
          
    def keyDown(keyName):
        action = 0
        if keyName == 0:
            action = env.actions.left
        elif keyName == 1:
            action = env.actions.right
        elif keyName == 2:
            action = env.actions.forward
        elif keyName == 3:
            action = env.actions.toggle
        elif keyName == 'RETURN':
            env.reset()
        elif keyName == 'ESCAPE':
            sys.exit(0)
        else:
            print("unknown key %s" % keyName)
            return

        
        return action

    
  
    

    with tf.Session() as sess:

        rl=[]
        rs=[] 
        for seed1 in range(10): 


            sess.run(init)
            for i in range(5000):
                env.seed(seed1)
                s=env.reset()
                #print ('i:%d'%(i))
              
                print('seedno:%d'%(seed1))
                rs=[]
                
                reward=0
                reward_im=0
                j=0
                print('i:%d'%(i))
                a=None
                    
                d=False
                while j<=200:
                    j=j+1
                      
                    temp=np.zeros([7,7,3,1])
                    temp[:,:,:,0]=s
                    hj2,output_p=sess.run([p,probs],feed_dict={data:temp})
                    #print(output_p)
                        
                        
                    an=np.random.choice(output_p[0],p=output_p[0])
                    if an==output_p[0,1]:
                        a=1
                    elif an==output_p[0,2]:
                        a=2
                    elif an==output_p[0,3]:
                        a=3
                    else:
                        a=0
                            
                    at=a
                      
                    a=keyDown(a)
                        
                    s1,r,d,_=env.step(a)
                    if i>4900:
                        env.render()
                    r1=modifyreward(s,s1,at)
                    if r1==0:
                        r1=r
                        

                    #print ('modr:%d'%(r1))
                    rs.append([s,s1,at,r1])
                        
                        
                    reward+=r
                    reward_im+=r1

                    s=s1
                    if d==True:
                        if reward>0:
                            for g in range(10):
                                z1=False      
                                d1=np.zeros([7,7,3,len(rs)])
                                d2=np.zeros([7,7,3,len(rs)])
                                d3=np.zeros([len(rs),4],dtype='int32')
                                d4=np.zeros([len(rs),1])
                                cnt=len(rs)
                                for l in range(cnt):
                                    d1[:,:,:,l]=rs[l][0]
                                    d2[:,:,:,l]=rs[l][1]
                                    d3[l,rs[l][2]]=1
                                    d4[l,:]=rs[l][3]
                                     
                                vs=sess.run(v,feed_dict={data:d1})
                                        
                                reward1=0
                                for k in reversed(range(len(rs))):  
                                    reward1=0.99*reward1+d4[k,0]
                                    d4[k,0]=reward1

                                adv=d4-vs
                                sess.run([optimize],feed_dict={data:d1,target_actions:d3,advantage:adv,target_v:d4,ent_constant:e})
                        else:
                            z1=False      
                            d1=np.zeros([7,7,3,len(rs)])
                            d2=np.zeros([7,7,3,len(rs)])
                            d3=np.zeros([len(rs),4],dtype='int32')
                            d4=np.zeros([len(rs),1])
                            cnt=len(rs)
                            for l in range(cnt):
                                d1[:,:,:,l]=rs[l][0]
                                d2[:,:,:,l]=rs[l][1]
                                d3[l,rs[l][2]]=1
                                d4[l,:]=rs[l][3]
                                     
                            vs=sess.run(v,feed_dict={data:d1})
                                        
                            reward1=0
                            for k in reversed(range(len(rs))):  
                                reward1=0.99*reward1+d4[k,0]
                                d4[k,0]=reward1

                            adv=d4-vs
                            sess.run([optimize],feed_dict={data:d1,target_actions:d3,advantage:adv,target_v:d4,ent_constant:e})



                        rl.append(reward)
                        re=np.array(rl)
                        print(reward)
                        print ('solved %d'%(countsuccess))
                        if reward>0:
                            countsuccess+=1
                            print ('solved!')
                        print (reward_im)
                            
                        
                        print('++++++++++++++++++')
                        break
                
                    
            save_path = saver.save(sess, "/home/aniket/Desktop/manas-task-phase/gym-minigrid/model2room%d.ckpt"%(seed1))



if __name__ == "__main__":
    main()

			
			
			
