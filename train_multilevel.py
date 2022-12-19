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
import re


# In[2]:


tf.keras.backend.clear_session()

_cfg = {
    "MIN_COUNT":1000,
    "INPUT_IMAGE_SIZE":512,
    "lr":0.01, "rbm-lr":0.01, 
    
    "BATCH_SIZE":128, "BUFFER_SIZE":128*3, 
    "model":"default", "DEBUG":False,
    "logdir":f"./tensorboard-logs/{datetime.isoformat(datetime.now())}",
    "ckpt_name":"default06",
}
try:
    cfg.update(_cfg)
except:
    cfg = _cfg.copy()


# In[3]:


# import pdb
# from IPython.core.debugger import set_trace


# In[4]:


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
        return unique_tag

    def load_common_tags(min_count = 1000):
        with open("./tags/tags000000000000.json", "r") as f :
            tags_d = [json.loads(line) for line in f.readlines()]

        df_tags = pd.DataFrame(tags_d)
        df_tags_0 = df_tags.query("category=='0'")
        df_tags_0["post_count"] = df_tags_0["post_count"].astype(int)
        df_tags_0 = df_tags_0.sort_values("post_count")
        df_tags_0_TOP = df_tags_0[df_tags_0["post_count"]>min_count]
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
    
    if "tag_structure" not in cfg:
        if not os.path.isfile("tag_structure.pkl"):
            load_tag_structure().to_pickle("tag_structure.pkl")
        cfg["tag_structure"] = pd.read_pickle("tag_structure.pkl")
        
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

        cache_path = f"metadata_processed_{cfg['MIN_COUNT']}.json.xz"
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
                        }
                        data += lzc.compress((json.dumps(tag_processed)+'\n').encode(encoding='utf-8'))
                data += lzc.flush()
                f.write(data)

        def metadata_gen(blocks):
            if blocks is None:
                blocks = []
            else :
                blocks = list(map(lambda x: str(x).zfill(3), blocks))
                
            def _metadata():
                with lzma.open(cache_path, mode='rb') as f:
                    for line in f:
                        yield line
                        
            for idx in map(lambda x: str(x).zfill(3), range(1000)):
                if idx not in blocks:
                    continue
                    
                block_cache_path = f"./metadata_processed/{cfg['MIN_COUNT']}_{idx}.json.xz"
                if not os.path.isfile(block_cache_path):
                    print(f"building cache {idx}")
                    metadata = _metadata()
                    with open(block_cache_path, 'wb', buffering=1024*1024) as f:
                        lzc = lzma.LZMACompressor()
                        data = b""
                        for x in metadata:
                            if re.match(b'{"id": "[0-9]*'+(idx.encode(encoding='utf-8', errors='strict'))+b'",', x):
                                data += lzc.compress(x)
                        data += lzc.flush()
                        f.write(data)
                
                # Yield
                result = {}
                with lzma.open(block_cache_path, mode='rb') as f:
                    for line in f:
                        info = json.loads(line)
                        result[info['id']]=info
                yield idx.zfill(4), result
        cfg["metadata_gen"]=metadata_gen
    return cfg


# In[5]:


class Datagen:
    def __init__(self, metadata_gen, N, train_test="train", verbose=True, normalize_label=False):
        self.verbose=verbose
        self.normalize_label=normalize_label
        self.metadata_gen = metadata_gen
        if train_test=="train":
            self.blocks = list(range(999))
        else:
            self.blocks = [999]
        self.encoder = tf.keras.layers.CategoryEncoding(output_mode="multi_hot", num_tokens=N)
        self.status = {"block":"", "id":"", "epoch":-1}
            
    def status(self):
        print(self.status)
        
    def _print(self, *args):
        if self.verbose:
            print(*args)

    def gen(self):
        blocks = self.metadata_gen(self.blocks)
        self.status["epoch"] += 1
        for block, metadata_dict in blocks:
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


def prepare_dataset(cfg, repeat=True, mode="all_tags"):
    if ("train_dataset" in cfg) and ("test_dataset" in cfg) : 
        return cfg
    metadata_gen, INPUT_IMAGE_SIZE = cfg["metadata_gen"], cfg["INPUT_IMAGE_SIZE"]
    BUFFER_SIZE, BATCH_SIZE = cfg["BUFFER_SIZE"], cfg["BATCH_SIZE"]
#     test_data_count = 1000
#     metadata_sort = sorted(metadata,key=lambda x: x["id"][-3:])
    
    reader_train = Datagen(metadata_gen, len(cfg[mode]), train_test = "train", verbose=False, normalize_label=False)
    reader_test = Datagen(metadata_gen, len(cfg[mode]), train_test = "test", verbose=False, normalize_label=False)
    cfg["reader_train"] = reader_train
    cfg["reader_test"] = reader_test

    output_signature=(
        tf.TensorSpec(shape=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(len(cfg[mode])), dtype=tf.float32),
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


# In[6]:


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


# In[7]:


def make_loss_function():
    structure = cfg["tag_structure"]#.groupby(3)["tag"].count()
    unique_tag_common = structure[structure["tag"].isin(cfg["visible_tags"])]
    unique_tag_common_visual = unique_tag_common[unique_tag_common[0]=="Visual characteristics"]

    def cluster(level=1, mask=unique_tag_common_visual.any(axis=1)):
        clusters = []
        for cls, cnt in unique_tag_common_visual[mask].groupby(level)["tag"].count().items():
            if (cnt > 100) and (level<4):
                sub_clusters = cluster(level+1, unique_tag_common_visual[level]==cls)
                misc = [[], level+1, 0]
                for clss, _level, _cnt in sub_clusters:
                    if _cnt > 30: 
                        clusters.append([clss, _level, _cnt])
                    else:
                        misc[0].extend(clss)
                        misc[2]+= _cnt
                if misc[2] > 30:
                    clusters.append(misc)
                elif len(misc[0])>0:
                    clusters[-1][0].extend(misc[0])
                    clusters[-1][2]+= misc[2]
            else : 
                clusters.append([[cls], level, cnt])
        return clusters
    class_cluster = pd.DataFrame(cluster())
    class_cluster_idx = class_cluster[0].map(lambda x : [{v:k for k,v in enumerate(cfg["all_tags"])}[y] for y in x])
    tags_cluster = class_cluster[0].map(lambda v : unique_tag_common_visual.loc[unique_tag_common_visual.loc[:,:5].isin(v).any(axis=1),"tag"].values)
    tags_cluster_idx = tags_cluster.map(lambda x : [{v:k for k,v in enumerate(cfg["all_tags"])}[y] for y in x])
    tags_cluster_idx_flat = np.unique(np.concatenate(tags_cluster_idx.values))
    assert(all(y<len(cfg["visible_tags"]) for y in tags_cluster_idx_flat))
    tags_misc_idx = [x for x in range(len(cfg["visible_tags"])) if (x not in tags_cluster_idx_flat)]
    class_itself_idx = [x for x in range(len(cfg["visible_tags"]), len(cfg["all_tags"]))]
    print(f"tags_cluster_idx:{len(tags_cluster_idx_flat)}" )
    print(f"tags_misc_idx:{len(tags_misc_idx)}" )
    assert(len(cfg["visible_tags"]) == len(tags_cluster_idx_flat) + len(tags_misc_idx))
    print(f"class_itself_idx:{len(class_itself_idx)}")
    assert(len(cfg["all_tags"]) - len(cfg["visible_tags"]) == len(class_itself_idx))

    encoder = tf.keras.layers.CategoryEncoding(output_mode="multi_hot", num_tokens=len(cfg["all_tags"]))
    classes_tags = [(encoder(c), encoder(t)) for c, t in zip(class_cluster_idx, tags_cluster_idx)]
    misc_tags = encoder(tags_misc_idx)
    class_itself = encoder(class_itself_idx)

    def _conv_kl(result, ans, mask):
        return tf.keras.losses.categorical_crossentropy(
            tf.multiply(result, mask),tf.multiply(ans, mask)
        )

    def conv_kl(true,pred): # convolution KL div
        result = tf.reduce_sum([
            # if true label does not contain the category of class, weight is limited to 0.1
            tf.clip_by_value(tf.reduce_max(tf.multiply(c,true),axis=-1),0.1,1)*
            _conv_kl(true,pred,t)
            for c, t in classes_tags
        ]+[
            # uncategorized weight 0.3
            tf.multiply(_conv_kl(true,pred,misc_tags), tf.constant(0.3))
        ]+[
            # category itself
            tf.multiply(_conv_kl(tf.nn.dropout(true,0.2),pred,class_itself), tf.constant(1.0))
        ], axis=0)
        print("loss true,pred,result shape", true.shape, pred.shape, result.shape)
        return result

    return conv_kl


# In[ ]:





# In[8]:


cfg = load_data(cfg)
cfg = prepare_dataset(cfg)


# In[9]:


tf.keras.backend.clear_session()

model = tf.keras.applications.EfficientNetB0(
    include_top=True,
    weights=None,
    input_tensor= tf.keras.layers.Resizing(256,256)(tf.keras.Input((512,512,3))),
    classes=len(cfg["all_tags"]),
    classifier_activation="sigmoid",
)
model.layers[-1].activity_regularizer=tf.keras.regularizers.L1(l1=1e-6)


# In[ ]:


cfg["lr"], cfg["rbm-lr"] = 0.001, 0.001
compile_model(cfg, options={"loss":make_loss_function()})
fit(model, cfg, 0, 50)


# In[ ]:


