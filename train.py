#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import lzma, tarfile
import tensorflow as tf
import numpy as np
from keras.layers import LeakyReLU


# In[2]:


## Parameters
# top tags
N = 1000
IMAGE_SIZE = 512


# In[3]:


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


# In[1]:


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
                    
tags_gen = get_tag()
with open('metadata_procesed.json.xz', 'wb', buffering=1024*1024) as f:
    lzc = lzma.LZMACompressor()
    data = b""
    for x in tags_gen:
        if int(x["score"]) > 5:
            tag_processed = {
                "id": x['id'],
                "pools": x["pools"],
                "tags_": [tags_topN_name.index(t["name"]) for t in x["tags"] if (t["name"] in tags_topN_name)]
            }
            data += lzc.compress((json.dumps(tag_processed)+'\n').encode(encoding='utf-8'))
    data += lzc.flush()
    f.write(data)


# In[38]:


# Define model
pretrained_model = tf.keras.applications.inception_v3.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(299, 299, 3),
    pooling=None,
    classes=N,
    classifier_activation='softmax'#LeakyReLU(alpha=0.05)
)

custom_model = tf.keras.applications.inception_v3.InceptionV3(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=(299, 299, 3),
    pooling=None,
    classes=N,
    classifier_activation=LeakyReLU(alpha=0.05)
)

for src, tgt in zip(pretrained_model.layers, custom_model.layers):
    if ("activation" not in src.name):
        try:
            assert (("_".join(src.name.split("_")[:-1])) == ("_".join(tgt.name.split("_")[:-1])))
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
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))


# In[30]:


with lzma.open('metadata_procesed.json.xz', mode='rb') as f:
    metadata = [json.loads(line) for line in f.readlines()]
files_lst = [f'./512px/0{x["id"].rjust(7,"0")[-3:]}/{x["id"]}.jpg' for x in metadata]
labels_lst = [x["tags_"] for x in metadata] 


# In[49]:


encoder = tf.keras.layers.CategoryEncoding(output_mode="multi_hot", num_tokens=N)
def gen():
    # TODO : shuffle
    for path, label in zip(files_lst[:100], labels_lst[:100]):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.uint8)
        label_enc = tf.keras.utils.normalize(encoder(label))
    yield image, label_end
    
dataset_train = tf.data.Dataset.from_generator(
    gen,
    output_signature=(
        tf.TensorSpec(shape=(512, 512, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(1000), dtype=tf.float32),
    )
)


# In[50]:


EPOCHS = 1 # debug
train_dataset = dataset_train.take(10).batch(1) # debug
# test_dataset = train_raw_dataset.take(10).batch(1) # debug

CORES_COUNT= 2
model_history = model.fit(train_dataset,
                          epochs=EPOCHS,
                          #validation_data=test_dataset,
                          use_multiprocessing=True,
                          workers=CORES_COUNT,
                          callbacks=[])#[DisplayCallback()])

#     print(" - Training finished, saving metrics into ./graphs")
#     save_model_history_metrics(EPOCHS, model_history)
#     print(" - Training finished, saving model into ./model")
#     output_path = "./model"
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#     model.save(output_path)
#     print(" - Model updated and saved")

