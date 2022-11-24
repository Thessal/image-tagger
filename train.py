#!/usr/bin/env python
# coding: utf-8

# In[3]:


import json
import lzma, tarfile
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
# from keras.layers import LeakyReLU
import os 
from datetime import datetime


# In[4]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
#             tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    except RuntimeError as e:
        print(e)


# In[5]:


## Parameters
# top tags
N = 1000
IMAGE_SIZE = 512
VERBOSE = False


# In[6]:


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


# In[7]:


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


# In[8]:


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


# In[14]:


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
#     classifier_activation=tfa.layers.Sparsemax()
    classifier_activation=tf.keras.Sequential([ tf.keras.layers.Softmax(), tf.keras.layers.ThresholdedReLU(theta=0.5), tf.keras.layers.Lambda(lambda x : x * 2) ])
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
# model = pretrained_model
## (B) Sparsemax
model = custom_model
####
## (A) Convolution resize
# trim_model = tf.keras.Model(inputs=model.layers[2].input, outputs=custom_model.output)
# result = trim_model(adapter_pooling(adapter_conv2d(adapter_input)))
## (B) Simple resize
result = model(adapter_resize(adapter_input))
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
                  tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5),
                  tf.keras.losses.Huber(delta=1.0),
              ],
             )

# tf.keras.utils.plot_model(model.layers[3])


# In[15]:


import requests
train_set = metadata

def _print(*args):
    if VERBOSE:
        print(*args)
    
encoder = tf.keras.layers.CategoryEncoding(output_mode="multi_hot", num_tokens=N)
def gengen(data_set):
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


# In[16]:


# Freeze convolution layers
def freeze(unfreeze=False):
    layer_count = 0
    for layer in model.layers[-1].layers:
        if layer.name.startswith("mixed"):
            layer_count += 1
        if layer.name.startswith("conv2d"):
            layer.trainable = True if unfreeze else False
        if layer_count > 8:
            # Do not freeze leaf convolution layer
            break


# In[17]:


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


freeze()
model.summary()
model_history = model.fit(train_dataset,
                          epochs=UNFREEZE_EPOCH,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_data=test_dataset,
                          validation_steps=1+STEPS_PER_EPOCH//10,
                          use_multiprocessing=True,
                          workers=CORES_COUNT,
                          callbacks=[DisplayCallback(), checkpoint, tensorboard_callback])


# In[ ]:


freeze(unfreeze=True)
model.summary()
model_history = model.fit(train_dataset,
                          initial_epoch=UNFREEZE_EPOCH,
                          epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_data=test_dataset,
                          validation_steps=1+STEPS_PER_EPOCH//10,
                          use_multiprocessing=True,
                          workers=CORES_COUNT,
                          callbacks=[DisplayCallback(), checkpoint, tensorboard_callback])


# In[ ]:


model.save(f"{output_path}/model_{model.history.epoch[-1]}_{datetime.isoformat(datetime.now())}")


# In[ ]:




