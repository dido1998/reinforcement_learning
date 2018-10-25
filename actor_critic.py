import tensorflow as tf 
import numpy as np 
import gym
from random import shuffle
tf.reset_default_graph()
data=tf.placeholder(tf.float32,shape=[None,4])
w1=tf.Variable(tf.truncated_normal([4,32]))
h1=tf.matmul(data,w1)
h1_=tf.nn.relu(h1)

w2=tf.Variable(tf.truncated_normal([32,1]))
output=tf.matmul(h1_,w2)
target_Q=tf.placeholder(tf.float32,[None,1])
loss=tf.reduce_mean(tf.square(target_Q-output))
optimize=tf.train.AdamOptimizer(0.01).minimize(loss)




w1a=tf.Variable(tf.truncated_normal([4,32]))
h1a=tf.matmul(data,w1a)
h1_a=tf.nn.relu(h1a)

w2a=tf.Variable(tf.truncated_normal([32,2]))
outputa=tf.matmul(h1_a,w2a)
probs=tf.nn.softmax(outputa)
target_a=tf.placeholder(tf.float32,shape=[None,2])
g_probs=tf.reduce_sum(tf.multiply(target_a,probs),reduction_indices=[1])
l_probs=tf.log(g_probs)
advantage=tf.placeholder(tf.float32,shape=[None,1])
lossa=-tf.reduce_sum(tf.multiply(advantage,l_probs))
optimizea=tf.train.AdamOptimizer(0.01).minimize(lossa)


#yo

init=tf.global_variables_initializer()
with tf.Session() as sess:
	env = gym.make('CartPole-v0')
	sess.run(init)
	p1=0.99
	
	
	rl=[]

	for i in range(4):
		
		s=env.reset()
		reward=0

		j=0
		a=None
		rs=[]
		d=False
		while j<330  :
			j=j+1
			s=np.reshape(s,[1,4])
			
			output_p=sess.run([probs],feed_dict={data:s})
			p=np.random.uniform(0,1)
			
				
			if p<p1:
				an=np.random.choice(output_p[0][0],p=output_p[0][0])
				if an==output_p[0][0,1]:
					a=1
				else:
					a=0
			else:
				a=np.argmax(output_p[0][0])
			
			s1,r,d,_=env.step(a)
			reward+=r
			s1=np.reshape(s1,[1,4])
			rs.append([s,s1,a,r])
			
					
		        s=s1
		        
			if d==True:
				shuffle(rs)
				d1=np.zeros([len(rs),4])
				d2=np.zeros([len(rs),4])
				d3=np.zeros([len(rs),2],dtype='int32')
				d4=np.zeros([len(rs),1])
				for l in range(len(rs)):
					d1[l,:]=rs[l][0]
					d2[l,:]=rs[l][1]
					d3[l,rs[l][2]]=1
					d4[l,:]=rs[l][3]
				vs=sess.run(output,feed_dict={data:d1})
				vs1=sess.run(output,feed_dict={data:d2})
				
				adv=d4+0.97*vs1-vs
				
				sess.run([optimizea],feed_dict={data:d1,advantage:adv,target_a:d3})
				reward1=0
				for k in reversed(range(len(rs))):

		 			
					reward1=0.80*reward1+d4[k,0]
					d4[k,0]=reward1
				w11=sess.run([w1,optimize],feed_dict={data:d1,target_Q:d4})
				print w1
				break
						
		"""if i%4==0:
			t1 = [rs.pop(random.randrange(len(rs))) for _ in xrange(8)]
			d1=np.zeros([len(t1),4])
			d2=np.zeros([len(t1),4])
			d3=np.zeros([len(t1),1],dtype='int32')
			d4=np.zeros([len(t1),1])
			for l in range(len(t1)):
				d1[l,:]=t1[l][0]
				d2[l,:]=t1[l][1]
				d3[l,:]=t1[l][2]
				d4[l,:]=t1[l][3]
			
			output1=sess.run([output],feed_dict={data:d1})
			output2=sess.run([output],feed_dict={data:d2})
			output1=output1[0]
			output2=output2[0]
			maxo2=np.max(output2,axis=1)
			output2=output1
			output2[:,d3[:,0]]=d4[:,0]+0.20*maxo2
			sess.run([optimize],feed_dict={data:d1,target_Q:output2,learning_rate:lr})"""
		
			
		
			

		

	        if i>20:
			if reward<20:
				sess.run(init)
		if i>100:
			if reward<50:
				sess.run(init)

		rl.append(reward)
		re=np.array(rl)
		print np.mean(re)
		p1=p1/(0.01*i+1)		
		print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
	




