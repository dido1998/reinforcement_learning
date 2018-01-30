import tensorflow as tf 
import numpy as np 
import gym
from random import shuffle

data=tf.placeholder(tf.float32,shape=[None,16])
w1=tf.Variable(tf.random_uniform([16,4],0.0,0.001))
output=tf.matmul(data,w1)


target_Q=tf.placeholder(tf.float32,[None,4])
loss=tf.reduce_mean(tf.square(target_Q-output))
learning_rate=tf.placeholder(tf.float32)
optimize=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


b=False
init=tf.global_variables_initializer()
with tf.Session() as sess:
	env = gym.make('FrozenLake-v0')
	sess.run(init)
	p1=0.50
	
	rs=[]
	rl=[]
	rep=[]
	for i in range(10000):
		
		s=env.reset()
		reward=0
		lr=0.001
		j=0
		a=None

		d=False
		while j<99 :
			j=j+1
			s=np.identity(16)[s:s+1]
			
			output_p=sess.run([output],feed_dict={data:s})
			#print output_p
			p=np.random.uniform(0,1)
			if p<p1:
				an=np.random.choice(output_p[0][0])
				if an==output_p[0][0,1]:
					a=1
				elif an==output_p[0][0,2]:
					a=2
				elif an==output_p[0][0,3]:
					a=3
				else:
					a=0
			else:
				a=np.argmax(output_p[0][0])
			
			s1,r,d,_=env.step(a)

                       
			reward+=r
			s2=s1

			s2=np.identity(16)[s2:s2+1]

			if r==0 and d==True:
				r=-100
				rs.append([s,s2,a,r])
				break
			elif r==1 and d==True:
				r=30;
				rs.append([s,s2,a,r])
				break
 
			
			"""print a
			print output_p
			print '___________'"""
			s2=s1

			s2=np.identity(16)[s2:s2+1]

			rs.append([s,s2,a,r])
			"""o=sess.run([output],feed_dict={data:s2})
			max_o=np.max(o[0],axis=1)
			target=output_p[0]
			target[0,a]=r+0.90*max_o
			sess.run([optimize],feed_dict={data:s,target_Q:target,learning_rate:lr})"""
					
		        s=s1
			
		if i%4==0:

			d1=np.zeros([len(rs),16])
			d2=np.zeros([len(rs),16])
			d3=np.zeros([len(rs),1],dtype='int32')
			d4=np.zeros([len(rs),1])
			for l in range(len(rs)):
				d1[l,:]=rs[l][0]
				d2[l,:]=rs[l][1]
				d3[l,:]=rs[l][2]
				d4[l,:]=rs[l][3]
			
			output1=sess.run([output],feed_dict={data:d1})
			output2=sess.run([output],feed_dict={data:d2})
			output1=output1[0]
			output2=output2[0]
			maxo2=np.max(output2,axis=1)
			output2=output1
			reward1=0
			
			#output2[:,d3[:,0]]=d4+0.90*maxo2
			reward1=0
			for k in reversed(range(len(rs))):
				reward1=0.50*reward1+d4[k,0]
				d4[k,0]=reward1
			output2[:,d3[:,0]]=d4
			sess.run([optimize],feed_dict={data:d1,target_Q:output2,learning_rate:lr})
			rs=[]
		
			
		
			

		

		
		print reward
		rl.append(reward)
		rep.append(reward)
		rh=np.array(rep)
		print np.mean(rh)
		
		re=np.array(rl)
		print np.mean(re)
		if i%100==0:
			if np.mean(rh)>0.75:
				print i
				break
			rep=[]

		
		print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
	        p1=p1/(0.0001*i+1)
		
		
			
			


