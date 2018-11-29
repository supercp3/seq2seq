import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt 
# print(tf.__version__)1.6.0

imdb=keras.datasets.imdb
(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)

# print("training entries:{},labels:{}".format(len(train_data),len(train_labels)))

# print(train_data[0])
# #每个样本都是一个整数数组，表示影评中的字词。每个标签都是0/1.
# #网络的输入需要长度一样，所以需要做处理
# print(train_labels[0])


word_index=imdb.get_word_index()

word_index={k:(v+3) for k,v in word_index.items()}
word_index['<PAD>']=0
word_index['<STAR>']=1
word_index['<UNK>']=2 #unknown
word_index['<UNUSED>']=3

reverse_word_index=dict([(value,key) for(key,value) in word_index.items()])
def decode_review(text):
	return ' '.join([reverse_word_index.get(i,'?') for i in text])
# print(decode_review(train_data[0]))
#转换为张量，馈送到神经网络
#填充数组，使其具有相同的长度max_length*num_reviews

#第一大模块处理数据
train_data=keras.preprocessing.sequence.pad_sequences(train_data,
	value=word_index['<PAD>'],
	padding="post",
	maxlen=256)

test_data=keras.preprocessing.sequence.pad_sequences(test_data,
	value=word_index['<PAD>'],
	padding='post',
	maxlen=256)

print(len(train_data[0]),len(train_data[1]))
# print(train_data[0])
# print(decode_review(train_data[0]))

#第二大模块构建模型
vocab_size=10000
model=keras.Sequential()
model.add(keras.layers.Embedding(vocab_size,16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16,activation=tf.nn.relu))
model.add(keras.layers.Dense(1,activation=tf.nn.sigmoid))
model.summary()

#配置模型以使用优化器和损失函数
model.compile(optimizer=tf.train.AdamOptimizer(),
	loss='binary_crossentropy',
	metrics=['accuracy'])

#创建验证集
x_val=train_data[:10000]
partial_x_train=train_data[10000:]

y_val=train_labels[:10000]
partial_y_train=train_labels[10000:]

#训练模型
history=model.fit(partial_x_train,
	partial_y_train,
	epochs=40,
	batch_size=512,
	validation_data=(x_val,y_val),
	verbose=1)

results=model.evaluate(test_data,test_labels)
print(results)

#history包含一个字典，包括训练期间发生的所有情况
history_dict=history.history
print(history_dict.keys())

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()