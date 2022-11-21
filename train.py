#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import lzma, tarfile
import tensorflow as tf
import numpy as np
from keras.layers import LeakyReLU
import os 


# In[2]:


## Parameters
# top tags
N = 1000
IMAGE_SIZE = 512

VERBOSE = False


# In[ ]:


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


# In[ ]:


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


# In[ ]:


# Define model
pretrained_model = tf.keras.applications.inception_v3.InceptionV3(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=(299, 299, 3),
    pooling=None,
    classes=N,
    classifier_activation='softmax'#LeakyReLU(alpha=0.05)
)

custom_model = tf.keras.applications.inception_v3.InceptionV3(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(299, 299, 3),
    pooling=None,
    classes=N,
    classifier_activation=LeakyReLU(alpha=0.05)
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
trim_model = tf.keras.Model(inputs=custom_model.layers[2].input, outputs=custom_model.output)
result = trim_model(adapter_pooling(adapter_conv2d(adapter_input)))

resized_model = tf.keras.Model(inputs=adapter_input, outputs=result)
model = resized_model
# model.summary()
# trim_model.summary()

model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=[
                  #tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy_sparse', dtype=None),
                  tf.keras.metrics.CategoricalAccuracy(name='accuracy', dtype=None)
              ],
             )
# sparse_softmax_cross_entropy_with_logits


# In[ ]:


import requests
# train_set = zip(files_lst[:100], labels_lst[:100])
# train_set = [(x["id"],x["tags_"]) for x in metadata[:100]]
train_set = metadata#[:10000]

def _print(*args):
    if VERBOSE:
        print(*args)
    
encoder = tf.keras.layers.CategoryEncoding(output_mode="multi_hot", num_tokens=N)
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

        image = tf.image.decode_image(image, channels=3, expand_animations=False, dtype=tf.uint8)
        label_enc = tf.keras.utils.normalize(encoder(label)) # note : use logit?
        yield image, tf.squeeze(label_enc)
    
dataset_train = tf.data.Dataset.from_generator(
    gen,
    output_signature=(
        tf.TensorSpec(shape=(512, 512, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(1000), dtype=tf.float32),
    )
)


# In[ ]:


#     train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(TRAINING_BATCH_SIZE)
#     train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
EPOCHS = 10 # debug
# train_dataset = dataset_train.take(10).batch(1) # debug
# test_dataset = dataset_train.take(10).batch(1) # debug
BUFFER_SIZE=64
TRAINING_BATCH_SIZE=32
CORES_COUNT= 2

# Checkpoints
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\n    - Training finished for epoch {}\n'.format(epoch + 1))
        # print(logs)
        output_path = "./model"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        with open (f"{output_path}/loss_{epoch}.json", "w") as f :
            f.write(json.dumps(logs))
        #model.save(output_path)                    
filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
logdir = "./tensorboard-logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# Train
# train_dataset = dataset_train.take(100).shuffle(BUFFER_SIZE).batch(TRAINING_BATCH_SIZE)
train_dataset = dataset_train.shuffle(BUFFER_SIZE).batch(TRAINING_BATCH_SIZE)
test_dataset = dataset_train.take(1000).batch(1000) # debug
model_history = model.fit(train_dataset,
                          epochs=EPOCHS,
                          validation_data=test_dataset,
                          use_multiprocessing=True,
                          workers=CORES_COUNT,
                          callbacks=[DisplayCallback(), checkpoint, tensorboard_callback])

