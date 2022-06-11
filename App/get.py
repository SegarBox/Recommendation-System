import os
import tensorflow as tf

def getRecommend(id_user):
	p = os.path.abspath('.')
	path = os.path.join(p, "Model")
	loaded = tf.saved_model.load(path)
	scores, titles = loaded([id_user])
	result=[]
	for x in titles[0][:10].numpy().tolist():
		result.append(x.decode("utf-8"))

	return result
