#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install tensorflow-probability==0.16.0
# !pip install tensorflow==2.8.0


# In[1]:


import json
import pandas as pd 
import urllib.parse
import numpy as np


# In[2]:


import json
import lzma, tarfile
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import os 
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import requests
import io
import pickle

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)


# In[3]:


tf.keras.backend.clear_session()

_cfg = {
    "N":5000, "MIN_COUNT":1000,
    "INPUT_IMAGE_SIZE":512,
    # "normalize_label":False,
    "lr":0.01, "rbm-lr":0.01, 
#     "RBM_N":200, "RBM_STEP":1,"rbm_regularization":False,
    
    "BATCH_SIZE":128, "BUFFER_SIZE":128*3, 
    "model":"default", "DEBUG":False,
    "logdir":f"./tensorboard-logs/{datetime.isoformat(datetime.now())}",
    "ckpt_name":"default06",
}
try:
    cfg.update(_cfg)
except:
    cfg = _cfg.copy()


# In[4]:


# import pdb
# from IPython.core.debugger import set_trace


# In[10]:


def load_data(cfg):
    def load_tag_structure():
        with open("e.json", "rt") as f:
            tag_groups = json.load(f)

        def traverse(info_lst, levels=[]):
            infos = []
            for info in info_lst:
                if type(info) != list:
                    continue
                elif all(type(x)==str for x in info):
                    if (len(info)==2) and (info[1].startswith('http')) :
                        infos.extend([info+levels])
                    else:
                        # print(info)
                        continue
                else:
                    level = [info[0]] if type(info[0]) == str else []
                    infos.extend(traverse(info, levels+level))
            return infos

        tag_enum = traverse(tag_groups)

        df_tag = pd.DataFrame(tag_enum)
        df_tag[1] = df_tag[1].apply(lambda x : urllib.parse.unquote(x).replace("https://danbooru.donmai.us/wiki_pages/",""))
        # df_tag[df_tag[3].astype(str).isin(["None",""])][2].unique()
        df_tag = df_tag[df_tag[2]!="Metatags"]
        df_tag = df_tag.astype(str)
        df_tag = df_tag[[2,3,4,5,6,1]]
        df_tag = df_tag[~ df_tag.apply(lambda x: x.apply(lambda y: "https://" in y).any(), axis=1)]
        df_tag = df_tag.applymap(lambda x : x.strip())
        df_tag = df_tag.apply(lambda x: pd.Series([x for x in x.to_list() if x != ""]), axis=1)
        df_tag = df_tag.fillna("")
        unique_tag = (df_tag+"/").cumsum(axis=1)[df_tag!=""].fillna("").applymap(lambda x: x[:-1])
        unique_tag["tag"] = df_tag[df_tag!=""].apply(lambda x: x.dropna().tolist()[-1], axis=1)
        # unique_tag[unique_tag[0] == "Visual characteristics"]
        return unique_tag

    def load_common_tags(min_count = 1000):
        with open("./tags/tags000000000000.json", "r") as f :
            tags_d = [json.loads(line) for line in f.readlines()]

        df_tags = pd.DataFrame(tags_d)
        df_tags_0 = df_tags.query("category=='0'")
        df_tags_0["post_count"] = df_tags_0["post_count"].astype(int)
        df_tags_0 = df_tags_0.sort_values("post_count")
        df_tags_0_TOP = df_tags_0[df_tags_0["post_count"]>min_count]
        # print(len(df_tags_0_TOP))
        return df_tags_0_TOP

    def common_categories_and_tags(structure, top_tags):
        unique_tag_common = structure[structure["tag"].isin(top_tags["name"])]
        unique_tag_common_visual = unique_tag_common[unique_tag_common[0]=="Visual characteristics"]

        level1_cats = unique_tag_common_visual[1].unique().tolist()
        level2_cats = unique_tag_common_visual[2].unique().tolist()
        level3_cats = unique_tag_common_visual[3].unique().tolist()
        level4_cats = unique_tag_common_visual[4].unique().tolist()
        level5_cats = unique_tag_common_visual[5][unique_tag_common_visual[5]!=""].unique().tolist()
        tag_to_cats_ = unique_tag_common_visual.set_index("tag")
        tag_to_cats = lambda x : ([y for y in np.unique(tag_to_cats_.loc[x,1:].values.ravel()) if (y!='')] if x in tag_to_cats_.index else []) + [x]
        all_tags = top_tags["name"].to_list() + level1_cats + level2_cats + level3_cats + level4_cats + level5_cats
        def make_label(tags):
            names = [x for tag in tags for x in tag_to_cats(tag)]
            idxs = [all_tags.index(x) for x in names]
            return idxs
        visible_tags = top_tags["name"].to_list()

        return all_tags, visible_tags, make_label

    if "all_tags" not in cfg:
        all_tags_path = f"all_tags_{cfg['MIN_COUNT']}.pkl"
        visible_tags_path = f"visible_tags_path{cfg['MIN_COUNT']}.pkl"
        if not (os.path.isfile(all_tags_path) and os.path.isfile(visible_tags_path)):
            structure, top_tags = load_tag_structure(), load_common_tags(min_count=cfg["MIN_COUNT"])
            cfg["all_tags"], cfg["visible_tags"], cfg["make_label"] = common_categories_and_tags(structure, top_tags)
            with open(all_tags_path ,'wb') as f:
                pickle.dump(cfg["all_tags"], f)
            with open(visible_tags_path ,'wb') as f:
                pickle.dump(cfg["visible_tags"], f)
        else:
            with open(all_tags_path ,'rb') as f:
                cfg["all_tags"] = pickle.load(f)
            with open(visible_tags_path ,'rb') as f:
                cfg["visible_tags"] = pickle.load(f)

    if "metadata" not in cfg:
        # Get image metadata
        # tags_tarinfo = ['metadata/2017/2017000000000001', 'metadata/2017/2017000000000002', 'metadata/2017/2017000000000003', 'metadata/2017/2017000000000004', 'metadata/2017/2017000000000005', 'metadata/2017/2017000000000006', 'metadata/2017/2017000000000007', 'metadata/2017/2017000000000008', 'metadata/2017/2017000000000009', 'metadata/2017/2017000000000010', 'metadata/2017/2017000000000011', 'metadata/2017/2017000000000012', 'metadata/2017/2017000000000013', 'metadata/2017/2017000000000014', 'metadata/2017/2017000000000015', 'metadata/2017/2017000000000016', 'metadata/2018/2018000000000000', 'metadata/2018/2018000000000001', 'metadata/2018/2018000000000002', 'metadata/2018/2018000000000003', 'metadata/2018/2018000000000004', 'metadata/2018/2018000000000005', 'metadata/2018/2018000000000006', 'metadata/2018/2018000000000007', 'metadata/2018/2018000000000008', 'metadata/2018/2018000000000009', 'metadata/2018/2018000000000010', 'metadata/2018/2018000000000011', 'metadata/2018/2018000000000012', 'metadata/2018/2018000000000013', 'metadata/2018/2018000000000014', 'metadata/2018/2018000000000015', 'metadata/2018/2018000000000016', 'metadata/2019/2019000000000000', 'metadata/2019/2019000000000001', 'metadata/2019/2019000000000002', 'metadata/2019/2019000000000003', 'metadata/2019/2019000000000004', 'metadata/2019/2019000000000005', 'metadata/2019/2019000000000006', 'metadata/2019/2019000000000007', 'metadata/2019/2019000000000008', 'metadata/2019/2019000000000009', 'metadata/2019/2019000000000010', 'metadata/2019/2019000000000011', 'metadata/2019/2019000000000012', 'metadata/2020/2020000000000000', 'metadata/2020/2020000000000001', 'metadata/2020/2020000000000002', 'metadata/2020/2020000000000003', 'metadata/2020/2020000000000004', 'metadata/2020/2020000000000005', 'metadata/2020/2020000000000006', 'metadata/2020/2020000000000007', 'metadata/2020/2020000000000008', 'metadata/2020/2020000000000009', 'metadata/2020/2020000000000010', 'metadata/2020/2020000000000011', 'metadata/2020/2020000000000012', 'metadata/2020/2020000000000013', 'metadata/2020/2020000000000014', 'metadata/2020/2020000000000015', 'metadata/2021-old/2021000000000000', 'metadata/2021-old/2021000000000001', 'metadata/2021-old/2021000000000002', 'metadata/2021-old/2021000000000003', 'metadata/2021-old/2021000000000004', 'metadata/2021-old/2021000000000005', 'metadata/2021-old/2021000000000006', 'metadata/2021-old/2021000000000007', 'metadata/2021-old/2021000000000008', 'metadata/2021-old/2021000000000009', 'metadata/2021-old/2021000000000010', 'metadata/2021-old/2021000000000011', 'metadata/2021-old/2021000000000012', 'metadata/2021-old/2021000000000013', 'metadata/2021-old/2021000000000014', 'metadata/2021-old/2021000000000015', 'metadata/2021-old/2021000000000016']
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

        cache_path = f"metadata_procesed_{cfg['MIN_COUNT']}.json.xz"
        if not os.path.isfile(cache_path):
            tags_gen = get_tag()
            with open(cache_path, 'wb', buffering=1024*1024) as f:
                lzc = lzma.LZMACompressor()
                data = b""
                for x in tags_gen:
                    if (int(x["score"]) > 5) and (x['file_ext'].lower() in ['jpg', 'jpeg', 'bmp', 'png', 'gif']):
                        tag_processed = {
                            "id": x['id'],
                            "pools": x["pools"],
                            "file_ext": x['file_ext'],
                            "tags_": cfg["make_label"]([t["name"] for t in x["tags"] if (t["name"] in cfg["visible_tags"])]),
#                             "tags_": [tags_topN_name.index(t["name"]) for t in x["tags"] if (t["name"] in cfg["visible_tags"])],
                        }
                        data += lzc.compress((json.dumps(tag_processed)+'\n').encode(encoding='utf-8'))
                data += lzc.flush()
                f.write(data)

        def metadata_gen(cache_path):
            with lzma.open(cache_path, mode='rb') as f:
                for line in f:
                    yield json.loads(line)
        cfg["metadata"]=metadata_gen(cache_path)
    return cfg


# In[26]:


class Datagen:
    def __init__(self, metadata, N, verbose=True, normalize_label=False):
        self.verbose=verbose
        self.normalize_label=normalize_label
        self.encoder = tf.keras.layers.CategoryEncoding(output_mode="multi_hot", num_tokens=N)
        self.blocks = self.block_gen()
#         self.blocks = {str(x).zfill(4):dict() for x in range(1000)}
#         for info in metadata:
#             # There are duplicate items with different value, Overwriting...
# #             if info['id'] in self.blocks[info["id"][-3:].zfill(4)]:
# #                 old = self.blocks[info['id'][-3:].zfill(4)][info['id']]
# #                 new = info
# #                 if str(old) != str(new):
# #                     print("Duplicates")
# #                     print(f"old : {old}")
# #                     print(f"new : {new}")
# #                     raise ValueError()
#             self.blocks[info["id"][-3:].zfill(4)][info['id']]=info
        self.status = {"block":"", "id":"", "epoch":-1}
    
    def block_gen(self):
        for i in range(1000):
            blk = {}
            idx = str(i).zfill(4)
            for info in metadata:
                if info["id"][-3:].zfill(4) == idx:
                    blk[info['id']]=info
            yield idx, blk
            
    def status(self):
        print(self.status)
        
    def _print(self, *args):
        if self.verbose:
            print(*args)

    def gen(self):
        self.status["epoch"] += 1
        for block, metadata_dict in self.blocks.items():
            self.status["block"] = block
            if len(metadata_dict)==0 : 
                continue
            self._print(f"Requesting block {block}")
            response = requests.get(f"http://192.168.20.50/danbooru2021/512px/{block}.tar")
            assert(response.status_code==200)
            self._print(f"Loaded block {block}")

            with io.BytesIO(response.content) as io_tar:
                with tarfile.open(fileobj=io_tar, mode='r|') as tar:
                    self._print(f"Opened block {block}")
                    for tarinfo in tar:
                        if tarinfo.isreg(): # regular file
                            img_id, img_ext = os.path.splitext(os.path.basename(tarinfo.name))
                            img_ext = img_ext.replace(".","")
                            self.status["id"] = img_id
                            if img_id not in metadata_dict:
                                self._print(f"{img_id} not found in metadata : {self.status}")
                                continue
                            info = metadata_dict[img_id]
                            img_id = info["id"]
                            img_ext = info["file_ext"]
                            label = info["tags_"]
                            if len(label) < 3 :
                                self._print(f"[error] {label}")
                                continue
                            with tar.extractfile(tarinfo) as f:
                                self._print(f"Read file {img_id, img_ext, label}")
                                try:
                                    if info["id"] in ['3928385']: 
                                        #{'block': '0385', 'id': '3928385'} jpeg::Uncompress failed. Invalid JPEG data or crop window. [Op:DecodeImage]
                                        continue
                                    image = tf.image.decode_image(f.read(), channels=3, expand_animations=False, dtype=tf.uint8)
                                    if self.normalize_label:
                                        label_enc = tf.keras.utils.normalize(self.encoder(label))
                                    else:
                                        label_enc = self.encoder(label)
                                    if (image.shape[0]!=512) or (image.shape[1]!=512):
                                        continue
                                    yield image, tf.squeeze(label_enc)
                                except:
                                    self._print(f"{info} : failed")


def prepare_dataset(cfg, repeat=True):
    if ("train_dataset" in cfg) and ("test_dataset" in cfg) : 
        return cfg
    metadata, N, INPUT_IMAGE_SIZE = cfg["metadata"], cfg["N"], cfg["INPUT_IMAGE_SIZE"]
    BUFFER_SIZE, BATCH_SIZE = cfg["BUFFER_SIZE"], cfg["BATCH_SIZE"]
    test_data_count = 1000
    metadata_sort = sorted(metadata,key=lambda x: x["id"][-3:])
    reader_train = Datagen(metadata_sort[:-test_data_count], N, verbose=False, normalize_label=False)
    reader_test = Datagen(metadata_sort[-test_data_count:], N, verbose=False, normalize_label=False)
    cfg["reader_train"] = reader_train
    cfg["reader_test"] = reader_test

    output_signature=(
        tf.TensorSpec(shape=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(N), dtype=tf.float32),
    )
    dataset_train = tf.data.Dataset.from_generator(reader_train.gen, output_signature=output_signature)
    dataset_test = tf.data.Dataset.from_generator(reader_test.gen, output_signature=output_signature)    
    if repeat:
        train_dataset = dataset_train.repeat().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        test_dataset = dataset_test.repeat().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    else:
        train_dataset = dataset_train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        test_dataset = dataset_test.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    cfg["train_dataset"] = train_dataset
    cfg["test_dataset"] = test_dataset
    return cfg

# cfg.pop("train_dataset")


# In[ ]:





# In[27]:


def compile_model(cfg, options={}):
    kwargs = {
        "optimizer" : 'SGD',
        "loss" : None,
        "metrics" : [
            tf.keras.metrics.KLDivergence(),
            tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            tfa.losses.SigmoidFocalCrossEntropy(),
            tfa.losses.SparsemaxLoss(),
            tf.keras.losses.MeanAbsoluteError(),
            tf.keras.losses.Huber(delta=1.0),
            tf.keras.losses.BinaryCrossentropy(),
            tf.keras.losses.BinaryFocalCrossentropy(),
        ],
    }
    if cfg["model"]=="default":
        kwargs["optimizer"] = tf.optimizers.Adam(learning_rate=cfg["lr"])
        kwargs["loss"] = tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO)
        kwargs.update(options)
    else:
        raise NotImplementedError()
        
    model.compile(**kwargs)
    
def fit(model, cfg, epoch_start, epoch_end):
    output_path = "./model"
    class DisplayCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print('\n    - Training finished for epoch {}\n'.format(epoch + 1))
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            print(logs)
    filepath=output_path+"/"+cfg['ckpt_name']+"-{epoch:02d}.ckpt"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, verbose=1, save_best_only=False)
    logdir = cfg["logdir"]
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    

    STEPS_PER_EPOCH=(2**15)//cfg["BATCH_SIZE"]
    model.fit(
        cfg["train_dataset"],
        epochs=epoch_end,
        initial_epoch=epoch_start,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=cfg["test_dataset"],
        validation_steps=1+STEPS_PER_EPOCH//10,
        use_multiprocessing=True,
        workers=2,
        callbacks=[DisplayCallback(), checkpoint, tensorboard_callback]
    )


# In[28]:


tf.keras.backend.clear_session()

model = tf.keras.applications.EfficientNetB0(
    include_top=True,
    weights=None,
    input_tensor= tf.keras.layers.Resizing(256,256)(tf.keras.Input((512,512,3))),
    classes=10000,
    classifier_activation="sigmoid",
)
model.layers[-1].activity_regularizer=tf.keras.regularizers.L1(l1=1e-6)


# In[29]:


cfg = load_data(cfg)


# In[30]:



cfg = prepare_dataset(cfg)
# print(cfg["all_tags"], cfg["visible_tags"], cfg["make_label"])

