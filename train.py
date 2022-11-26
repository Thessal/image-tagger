#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tensorflow-probability==0.16.0


# In[2]:


import json
import lzma, tarfile
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
# from keras.layers import LeakyReLU
import os 
from datetime import datetime

import matplotlib.pyplot as plt
import tensorflow_probability as tfp


# In[3]:



# tf.config.set_visible_devices([], 'GPU')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
#             tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    except RuntimeError as e:
        print(e)


# In[4]:


## Parameters
# top tags
N = 1000
IMAGE_SIZE = 512
VERBOSE = False


# In[5]:


# Get tag list
with open("./tags/tags000000000000.json", "r") as f :
    tags_d = [json.loads(line) for line in f.readlines()]
tags_top = sorted(
    [x for x in tags_d if x["category"]=="0"], 
    key=lambda x: int(x["post_count"]),
    reverse=True
)
tags_topN = [x for x in tags_top[:N]]
tags_topN_name = sorted([x["name"] for x in tags_topN])
# sorted([x for x in tags_d if ("_chart" in x["name"]) and x['post_count']!='0'], key=lambda x:int(x["post_count"]), reverse=True)


# In[6]:


# Get image metadata
tags_tarinfo = ['metadata/2017/2017000000000001', 'metadata/2017/2017000000000002', 'metadata/2017/2017000000000003', 'metadata/2017/2017000000000004', 'metadata/2017/2017000000000005', 'metadata/2017/2017000000000006', 'metadata/2017/2017000000000007', 'metadata/2017/2017000000000008', 'metadata/2017/2017000000000009', 'metadata/2017/2017000000000010', 'metadata/2017/2017000000000011', 'metadata/2017/2017000000000012', 'metadata/2017/2017000000000013', 'metadata/2017/2017000000000014', 'metadata/2017/2017000000000015', 'metadata/2017/2017000000000016', 'metadata/2018/2018000000000000', 'metadata/2018/2018000000000001', 'metadata/2018/2018000000000002', 'metadata/2018/2018000000000003', 'metadata/2018/2018000000000004', 'metadata/2018/2018000000000005', 'metadata/2018/2018000000000006', 'metadata/2018/2018000000000007', 'metadata/2018/2018000000000008', 'metadata/2018/2018000000000009', 'metadata/2018/2018000000000010', 'metadata/2018/2018000000000011', 'metadata/2018/2018000000000012', 'metadata/2018/2018000000000013', 'metadata/2018/2018000000000014', 'metadata/2018/2018000000000015', 'metadata/2018/2018000000000016', 'metadata/2019/2019000000000000', 'metadata/2019/2019000000000001', 'metadata/2019/2019000000000002', 'metadata/2019/2019000000000003', 'metadata/2019/2019000000000004', 'metadata/2019/2019000000000005', 'metadata/2019/2019000000000006', 'metadata/2019/2019000000000007', 'metadata/2019/2019000000000008', 'metadata/2019/2019000000000009', 'metadata/2019/2019000000000010', 'metadata/2019/2019000000000011', 'metadata/2019/2019000000000012', 'metadata/2020/2020000000000000', 'metadata/2020/2020000000000001', 'metadata/2020/2020000000000002', 'metadata/2020/2020000000000003', 'metadata/2020/2020000000000004', 'metadata/2020/2020000000000005', 'metadata/2020/2020000000000006', 'metadata/2020/2020000000000007', 'metadata/2020/2020000000000008', 'metadata/2020/2020000000000009', 'metadata/2020/2020000000000010', 'metadata/2020/2020000000000011', 'metadata/2020/2020000000000012', 'metadata/2020/2020000000000013', 'metadata/2020/2020000000000014', 'metadata/2020/2020000000000015', 'metadata/2021-old/2021000000000000', 'metadata/2021-old/2021000000000001', 'metadata/2021-old/2021000000000002', 'metadata/2021-old/2021000000000003', 'metadata/2021-old/2021000000000004', 'metadata/2021-old/2021000000000005', 'metadata/2021-old/2021000000000006', 'metadata/2021-old/2021000000000007', 'metadata/2021-old/2021000000000008', 'metadata/2021-old/2021000000000009', 'metadata/2021-old/2021000000000010', 'metadata/2021-old/2021000000000011', 'metadata/2021-old/2021000000000012', 'metadata/2021-old/2021000000000013', 'metadata/2021-old/2021000000000014', 'metadata/2021-old/2021000000000015', 'metadata/2021-old/2021000000000016']
def get_tag():
    tags_lst = {}
    with tarfile.open(name='./tags/metadata.json.tar.xz', mode='r|xz') as tar:
        # tarinfo = tar.next()
        for tarinfo in tar:
            print(tarinfo.name)
            if tarinfo.isreg(): # regular file
                tags_lst[tarinfo.name] = []
                with tar.extractfile(tarinfo) as f:
                    for line in f.readlines():
                        try :
                            result = json.loads(line)
                            yield result
                        except Exception as e:
                            print(f"[json load] {e} : {line}")

if not os.path.isfile("metadata_procesed.json.xz"):
    tags_gen = get_tag()
    with open('metadata_procesed.json.xz', 'wb', buffering=1024*1024) as f:
        lzc = lzma.LZMACompressor()
        data = b""
        for x in tags_gen:
            if (int(x["score"]) > 5) and (x['file_ext'].lower() in ['jpg', 'jpeg', 'bmp', 'png', 'gif']):
                tag_processed = {
                    "id": x['id'],
                    "pools": x["pools"],
                    "file_ext": x['file_ext'],
                    "tags_": [tags_topN_name.index(t["name"]) for t in x["tags"] if (t["name"] in tags_topN_name)],
                }
                data += lzc.compress((json.dumps(tag_processed)+'\n').encode(encoding='utf-8'))
        data += lzc.flush()
        f.write(data)

with lzma.open('metadata_procesed.json.xz', mode='rb') as f:
    metadata = [json.loads(line) for line in f.readlines()]


# In[7]:


# # Define model
# pretrained_model = tf.keras.applications.inception_v3.InceptionV3(
#     include_top=True,
#     weights='imagenet',
#     input_tensor=None,
#     input_shape=(299, 299, 3),
#     pooling=None,
#     classes=N,
#     classifier_activation='softmax'
# )

# custom_model = tf.keras.applications.inception_v3.InceptionV3(
#     include_top=True,
#     weights=None,
#     input_tensor=None,
#     input_shape=(299, 299, 3),
#     pooling=None,
#     classes=N,
#     classifier_activation=tfa.layers.Sparsemax()
# )

# getname = lambda s : s[:len(s)-1-(s)[::-1].find("_")] if s[-1].isnumeric() else s
# for src, tgt in zip(pretrained_model.layers, custom_model.layers):
#     if ("activation" not in src.name):
#         try:
#             assert (getname(src.name) == getname(tgt.name))
#         except:
#             print(src.name, tgt.name)
#             raise Exception
#         tgt.set_weights(src.get_weights())

# adapter_input = tf.keras.Input((512,512,3))
# adapter_conv2d = tf.keras.layers.Conv2D(strides=(3,3),filters=32,padding='valid',kernel_size=(16,16))
# adapter_pooling = tf.keras.layers.MaxPooling2D(pool_size=(18, 18), strides=(1, 1), padding='valid')
# trim_model = tf.keras.Model(inputs=custom_model.layers[2].input, outputs=custom_model.output)
# result = trim_model(adapter_pooling(adapter_conv2d(adapter_input)))

# resized_model = tf.keras.Model(inputs=adapter_input, outputs=result)
# model = resized_model

# model.compile(optimizer='adam',
# #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#               loss=tfa.losses.SigmoidFocalCrossEntropy(),
#               metrics=[
#                   #tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy_sparse', dtype=None),
#                   tf.keras.metrics.KLDivergence(),
#                   tf.keras.metrics.CategoricalAccuracy(name='accuracy', dtype=None),
#                   tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#                   tfa.losses.SigmoidFocalCrossEntropy(),
#                   tfa.losses.SparsemaxLoss(), # logits
#               ],
#              )
# # sparse_softmax_cross_entropy_with_logits


# In[8]:



# activation = #tf.keras.layers.Lambda(lambda tensor : tf.keras.layers.ThresholdedReLU(theta=0.5)(tf.keras.layers.Softmax()(tensor))*2 )
def activation_function(tensor):
    prob = tf.keras.activations.softmax(tensor)
    scale = tf.reduce_max(prob) 
    thres_max = tfp.stats.percentile(prob, q=80.)
    thres_min = tfp.stats.percentile(prob, q=20.)
    saturate = 0.2
    averse = 10
    return prob + (saturate-1) * tf.clip_by_value(prob, thres_max, scale) + (averse-1) * tf.clip_by_value(prob, -thres_max, thres_min)

# activation = tf.keras.layers.Lambda(lambda tensor : tf.keras.activations.relu(10*tf.keras.activations.softmax(tensor)) )
activation = tf.keras.layers.Lambda(activation_function)
test_x = np.arange(-1,2,0.1)[np.newaxis,:]
test_y = activation(tf.convert_to_tensor(test_x)).numpy()
plt.plot(test_x[0],test_y[0])


# In[9]:


def attention_reshape(image_batch):
    # Reshapes input image
    assert image_batch.dtype==tf.float32
    patches = tf.image.extract_patches(
                images=image_batch,
                sizes=[1, 149, 149, 1],
                strides=[1, 121, 121, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )
    patches = tf.reshape(patches,(-1,4,4,149,149,3))
    attention = tf.keras.layers.MultiHeadAttention(
        num_heads=4,
        key_dim=3,
        value_dim=None,
        dropout=0.0,
        use_bias=True,
        output_shape=2,
        attention_axes=(1,2),
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
    )
    self_attention = attention(patches,patches)
    self_attention_stack = tf.reshape(tf.einsum('bYXyxc->byxYXc', self_attention), (10,149,149,32))
    return self_attention_stack


# In[10]:


def patch_model(model, adapter_input):
    # Divide and conquer
    adapter_input_resize = tf.keras.layers.Resizing(1024,1024)(adapter_input) # FIXME
    embed = tf.keras.Model(inputs=model.input, outputs=model.layers[-1].input)
    patches = tf.image.extract_patches(
                images=adapter_input_resize,
                sizes=[1, 299, 299, 1], # model input size
                strides=[1, 121, 121, 1], # patch size
                rates=[1, 1, 1, 1],
                padding="VALID",
            )
    patches = tf.reshape(patches,(-1,299,299,3))
    vectors = tf.reshape(embed(patches), (-1,6*6,2048))
    attention = tf.keras.layers.MultiHeadAttention(
        num_heads=6,
        key_dim=3, # TODO : Tune
        value_dim=None,
        dropout=0.0,
        use_bias=True,
        output_shape=None,
        attention_axes=(1),
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
    )
    self_attention = attention(vectors,vectors)
    result = model.layers[-1](tf.keras.layers.AveragePooling1D(pool_size=6*6,data_format="channels_last")(self_attention))
    result = tf.einsum("abc->ac", result)
    return result


# In[11]:


# Define model
pretrained_model = tf.keras.applications.inception_v3.InceptionV3(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=(299, 299, 3),
    pooling=None,
    classes=N,
    classifier_activation='softmax'
)

custom_model = tf.keras.applications.inception_v3.InceptionV3(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(299, 299, 3),
    pooling=None,
    classes=N,
    classifier_activation=activation,
    # Or, tf.keras.layers.Lambda(lambda tensor : tf.keras.layers.ThresholdedReLU(theta=0.5)(tf.keras.layers.Softmax()(tensor))*2 )
)

getname = lambda s : s[:len(s)-1-(s)[::-1].find("_")] if s[-1].isnumeric() else s
for src, tgt in zip(pretrained_model.layers, custom_model.layers):
    if ("activation" not in src.name):
        try:
            assert (getname(src.name) == getname(tgt.name))
        except:
            print(src.name, tgt.name)
            raise Exception
        tgt.set_weights(src.get_weights())

adapter_input = tf.keras.Input((512,512,3))
adapter_conv2d = tf.keras.layers.Conv2D(strides=(3,3),filters=32,padding='valid',kernel_size=(16,16))
adapter_pooling = tf.keras.layers.MaxPooling2D(pool_size=(18, 18), strides=(1, 1), padding='valid')
adapter_resize = tf.keras.layers.Resizing(299,299)

####
## (A) softmax
base_model = pretrained_model
## (B) Sparsemax
# base_model = custom_model
####
## (A) Convolution resize
# trim_model = tf.keras.Model(inputs=base_model.layers[2].input, outputs=base_model.output)
# result = trim_model(adapter_pooling(adapter_conv2d(adapter_input)))
## (B) Simple resize
result = base_model(adapter_resize(adapter_input))
## (C) Attention resize
# trim_model = tf.keras.Model(inputs=base_model.layers[2].input, outputs=base_model.output)
# result = trim_model(attention_reshape(adapter_input))
## (D) Patch attention
# result = patch_model(base_model, adapter_input)
####

model = tf.keras.Model(inputs=adapter_input, outputs=result)
model.compile(optimizer='adam',
              loss=tfa.losses.SigmoidFocalCrossEntropy(),
              metrics=[
                  tf.keras.metrics.KLDivergence(),
                  tf.keras.metrics.CategoricalAccuracy(name='accuracy', dtype=None),
                  tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  tfa.losses.SigmoidFocalCrossEntropy(),
                  tfa.losses.SparsemaxLoss(),
                  tf.keras.losses.MeanAbsoluteError(),
#                   tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5),
                  tf.keras.losses.Huber(delta=1.0),
              ],
             )

# tf.keras.utils.plot_model(model.layers[3])


# In[ ]:





# In[12]:


import requests
train_set = metadata

def _print(*args):
    if VERBOSE:
        print(*args)
    
encoder = tf.keras.layers.CategoryEncoding(output_mode="multi_hot", num_tokens=N)
def gengen(data_set):
    """
    def gen_block():
        # Reduces fs overhead
        blocks = [f'http://192.168.20.50/blocks.txt']
        for block in blocks
            block_tar = block
            with tarfile.open(io_block_tar, mode='r|xz') as tar:
                for tarinfo in tar:
                    #print(tarinfo.name)
                    if tarinfo.isreg(): # regular file
                        img_id, img_ext = os.path.split(tarinfo.name)
                        info = train_dict[img_id]
                        
                        img_id = info["id"]
                        img_ext = info["file_ext"]
                        label = info["tags_"]
                        if len(label) < 3 :
                            _print(f"[error] {label}")
                            continue
                        
                        
                        with tar.extractfile(tarinfo) as f:
                            image = tf.io.read_file(path)
                            image = tf.image.decode_image(image, channels=3, expand_animations=False, dtype=tf.uint8)
                            label_enc = tf.keras.utils.normalize(encoder(label)) # note : use logit?
                            yield image, tf.squeeze(label_enc)
    """
    def gen():
        # TODO : shuffle
        for info in train_set:
            img_id = info["id"]
            img_ext = info["file_ext"]
            label = info["tags_"]
            if len(label) < 3 :
                _print(f"[error] {label}")
                continue
                
            path = f'http://192.168.20.50/danbooru2021/512px/0{img_id.rjust(7,"0")[-3:]}/{img_id}.{img_ext}'
            try:
                response = requests.get(path)
                image = response.content
                if response.status_code != 200:
                    _print(f"[error] {path}")
                    continue        
                if False:
                    path = f'./512px/0{img_id.rjust(7,"0")[-3:]}/{img_id}.{img_ext}'
                    if not os.path.exists(path):
                        _print(f"[error] {path}")
                        continue
                    image = tf.io.read_file(path)
            except:
                print(f"[Error] {path}")

            image = tf.image.decode_image(image, channels=3, expand_animations=False, dtype=tf.uint8)
            label_enc = tf.keras.utils.normalize(encoder(label)) # note : use logit?
            yield image, tf.squeeze(label_enc)
    return gen
    
output_signature=(
    tf.TensorSpec(shape=(512, 512, 3), dtype=tf.uint8),
    tf.TensorSpec(shape=(1000), dtype=tf.float32),
)
dataset_train = tf.data.Dataset.from_generator( gengen(metadata[:-1000]), output_signature=output_signature)
dataset_test = tf.data.Dataset.from_generator( gengen(metadata[-1000:]),output_signature=output_signature)


# In[66]:


# # tmp = dataset_test.take(1)
# tmp.get_single_element()


# In[13]:


# def freeze(model, unfreeze=False):
#     # Freeze convolution layers only
#     for layer in model.layers:
#         if 'layers' in layer.__dict__:
#             # TODO : avoid duplicate freeze for same layer
#             freeze(layer.layers, unfreeze)
#         else:
#             if layer.name.startswith("conv2d"):
#                 layer.trainable = True if unfreeze else False

                
# Freeze convolution layers
def freeze(unfreeze=False):
    layer_count = 0
    for layer in model.layers[-1].layers:
        if layer.name.startswith("mixed"):
            layer_count += 1
        if layer.name.startswith("conv2d"):
            layer.trainable = True if unfreeze else False
            ##Optional
#         if layer_count > 8:
#             # Do not freeze leaf convolution layer
#             break


# In[14]:


TRAINING_BATCH_SIZE=128
BUFFER_SIZE=TRAINING_BATCH_SIZE*3
STEPS_PER_EPOCH=(2**15)//TRAINING_BATCH_SIZE
CORES_COUNT= 2
EPOCHS = 3 * len(train_set) // TRAINING_BATCH_SIZE // STEPS_PER_EPOCH
UNFREEZE_EPOCH = len(train_set) // TRAINING_BATCH_SIZE // STEPS_PER_EPOCH // 30

# Checkpoints
output_path = "./model"
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\n    - Training finished for epoch {}\n'.format(epoch + 1))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        with open (f"{output_path}/loss_{epoch}.json", "w") as f :
            f.write(json.dumps(logs))           
filepath=output_path+"/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
# checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_kullback_leibler_divergence', verbose=1, save_best_only=True, mode='min')
logdir = f"./tensorboard-logs/{datetime.isoformat(datetime.now())}"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

train_dataset = dataset_train.repeat().shuffle(BUFFER_SIZE).batch(TRAINING_BATCH_SIZE)#.cache()
test_dataset = dataset_test.repeat().shuffle(BUFFER_SIZE).batch(TRAINING_BATCH_SIZE)#.cache()


# In[ ]:


freeze(base_model)
model.summary()
model_history = model.fit(train_dataset,
                          epochs=UNFREEZE_EPOCH,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_data=test_dataset,
                          validation_steps=1+STEPS_PER_EPOCH//10,
                          use_multiprocessing=True,
                          workers=CORES_COUNT,
                          callbacks=[DisplayCallback(), checkpoint, tensorboard_callback])


# In[18]:


# model.save(f"{output_path}/model_{datetime.isoformat(datetime.now())}_{model.history.epoch[-1]}")
# model.save_weights("./weights.hdf5")
# model.save_weights("./model/weights-improvement-32-0.27.hdf5")
# dataset_train = tf.data.Dataset.from_generator( gengen(metadata[256*22*128:-1000]), output_signature=output_signature)


# In[20]:


freeze(base_model,unfreeze=True)
model.summary()
model_history = model.fit(train_dataset,
                          initial_epoch=UNFREEZE_EPOCH,
                          epochs=31,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_data=test_dataset,
                          validation_steps=1+STEPS_PER_EPOCH//10,
                          use_multiprocessing=True,
                          workers=CORES_COUNT,
                          callbacks=[DisplayCallback(), checkpoint, tensorboard_callback])


# In[ ]:


# @keras_export("keras.metrics.TopKCategoricalAccuracy")
class HeadKCategoricalCrossEntropy(tf.keras.metrics.MeanMetricWrapper):
#     @dtensor_utils.inject_mesh
    def __init__(self, k=200, name="head_k_categorical_crossentorpy", dtype=None):
        super().__init__(
            lambda yt, yp: tf.keras.metrics.categorical_crossentropy(
                yt[:,:k], yp[:,:k]
            ),
            name,
            dtype=dtype,
#             k=k,
        )
            
class WeightKCategoricalCrossEntropy(tf.keras.metrics.MeanMetricWrapper):
#     @dtensor_utils.inject_mesh
    def __init__(self, name="weighted_k_categorical_crossentorpy", dtype=None):
        # Weight on starting 200 element of 1000 element
        sigm = lambda x : 1 / ( 1 +np.exp(0.02*(-x+800)))
        self.weight=sigm(np.arange(1000,0,-1))
        super().__init__(
            lambda yt, yp: tf.keras.metrics.categorical_crossentropy(
                yt*self.weight, yp*self.weight
            ),
            name,
            dtype=dtype,
        )

model.compile(optimizer='nadam', # SGD
              loss=tfa.losses.SparsemaxLoss(),
              metrics=[
                  tf.keras.metrics.KLDivergence(),
                  #tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  WeightKCategoricalCrossEntropy(name='accuracy'), # TODO : rename 'accuracy'
                  HeadKCategoricalCrossEntropy(k=200),
                  tfa.losses.SigmoidFocalCrossEntropy(),
                  tfa.losses.SparsemaxLoss(),
                  tf.keras.losses.MeanAbsoluteError(),
                  tf.keras.losses.Huber(delta=1.0),
              ],
             )
model.summary()
model_history = model.fit(train_dataset,
                          initial_epoch=31,
                          epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_data=test_dataset,
                          validation_steps=1+STEPS_PER_EPOCH//10,
                          use_multiprocessing=True,
                          workers=CORES_COUNT,
                          callbacks=[DisplayCallback(), checkpoint, tensorboard_callback])


# In[ ]:


model.save_weights("./weights.hdf5")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


print("test")


# In[ ]:




