import tensorflow as tf 
import numpy as np 
import gym
import random
tf.reset_default_graph()
data=tf.placeholder(tf.float32,shape=[None,4])
w1=tf.Variable(tf.truncated_normal([4,2]))
output=tf.matmul(data,w1)


target_Q=tf.placeholder(tf.float32,[None,2])
loss=tf.reduce_mean(tf.square(target_Q-output))
learning_rate=tf.placeholder(tf.float32)
optimize=tf.train.AdamOptimizer(learning_rate).minimize(loss)



init=tf.global_variables_initializer()
with tf.Session() as sess:
	env = gym.make('CartPole-v0')
	sess.run(init)
	p1=0.85
	
	rs=[]
	rl=[]

	for i in range(3000):
		print 
		s=env.reset()
		reward=0
		lr=1e-24
		j=0
		a=None

		d=False
		while j<330 and d==False :
			j=j+1
			s=np.reshape(s,[1,4])
			
			output_p=sess.run([output],feed_dict={data:s})

			p=np.random.uniform(0,1)
			if p<p1:
				an=np.random.choice(output_p[0][0])
				if an==output_p[0][0,1]:
					a=1
				else:
					a=0
			else:
				a=np.argmax(output_p[0][0])
			
			s1,r,d,_=env.step(a)
			env.render()
			reward+=r
			
			"""print a
			print output_p
			print '___________'"""
			rs.append([s,s1,a,r])
			s1=np.reshape(s1,[1,4])
			o=sess.run([output],feed_dict={data:s1})
			max_o=np.max(o[0],axis=1)
			target=output_p[0]
			target[0,a]=r+0.20*max_o
			sess.run([optimize],feed_dict={data:s,target_Q:target,learning_rate:lr})
					
		        s=s1
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

		if reward==200:
			print i
			break
		print reward
		rl.append(reward)
		re=np.array(rl)
		print np.mean(re)
		p1=p1/(0.01*i+1)		
		print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
	




