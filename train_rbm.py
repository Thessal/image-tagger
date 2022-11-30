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
import os 
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import requests
import io

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

cfg = {
    "N":2000, "INPUT_IMAGE_SIZE":512, 
    "BATCH_SIZE":128, "BUFFER_SIZE":128*3, 
    "model":"RBM", "DEBUG":False
}


# In[3]:


import pdb
from IPython.core.debugger import set_trace
tf.keras.backend.clear_session()


# In[4]:


def load_data(cfg):
    N = cfg["N"]
    if "tags_topN_name" not in cfg:
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
        cfg["tags_topN_name"]=tags_topN_name

    if "metadata" not in cfg:
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

        cache_path = f"metadata_procesed_TOP{N}.json.xz"
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
                            "tags_": [tags_topN_name.index(t["name"]) for t in x["tags"] if (t["name"] in tags_topN_name)],
                        }
                        data += lzc.compress((json.dumps(tag_processed)+'\n').encode(encoding='utf-8'))
                data += lzc.flush()
                f.write(data)

        with lzma.open(cache_path, mode='rb') as f:
            metadata = [json.loads(line) for line in f.readlines()]
        cfg["metadata"]=metadata
    return cfg


# In[5]:


class Datagen:
    def __init__(self, metadata, N, verbose=True, normalize_label=False):
        self.verbose=verbose
        self.normalize_label=normalize_label
        self.encoder = tf.keras.layers.CategoryEncoding(output_mode="multi_hot", num_tokens=N)
        self.blocks = {str(x).zfill(4):dict() for x in range(1000)}
        for info in metadata:
            self.blocks[info["id"][-3:].zfill(4)][info['id']]=info
        self.status = {"block":"", "id":""}
            
    def status(self):
        print(self.status)
        
    def _print(self, *args):
        if self.verbose:
            print(*args)

    def gen(self):
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
                                image = tf.image.decode_image(f.read(), channels=3, expand_animations=False, dtype=tf.uint8)
                                if self.normalize_label:
                                    label_enc = tf.keras.utils.normalize(encoder(label))
                                else:
                                    label_enc = self.encoder(label)
                                yield image, tf.squeeze(label_enc)


def prepare_dataset(cfg, repeat=True):
    if ("train_dataset" in cfg) and ("test_dataset" in cfg) : 
        return cfg
    metadata, N, INPUT_IMAGE_SIZE = cfg["metadata"], cfg["N"], cfg["INPUT_IMAGE_SIZE"]
    BUFFER_SIZE, BATCH_SIZE = cfg["BUFFER_SIZE"], cfg["BATCH_SIZE"]
    test_data_count = 1000
    metadata_sort = sorted(metadata,key=lambda x: x["id"][-3:])
    reader_train = Datagen(metadata_sort[:-test_data_count], N, verbose=False)
    reader_test = Datagen(metadata_sort[-test_data_count:], N, verbose=False)
#     reader_train = Datagen(metadata_sort[:-test_data_count], N, verbose=True)
#     reader_test = Datagen(metadata_sort[-test_data_count:], N, verbose=True)

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


# In[6]:


def load_pretrained_model_01():
    # tf.keras.models.save_model(model, "./model.hdf5")
    # tf.keras.models.load_model("./model.hdf5")
    # !tensorflowjs_converter --input_format=keras ./model_tfjs.h5 /tfjs_model

    base_model = tf.keras.applications.inception_v3.InceptionV3(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=(299, 299, 3),
        classes=1000,
        classifier_activation='softmax'
    )

    # load pretrained model
    adapter_input = tf.keras.Input((512,512,3))
    adapter_resize = tf.keras.layers.Resizing(299,299)
    result = base_model(adapter_resize(adapter_input))
    model = tf.keras.Model(inputs=adapter_input, outputs=result)
    model.load_weights("./model/weights-improvement-145-2.71.hdf5")
    return model


# In[ ]:





# In[ ]:





# In[7]:


class Rbm(tf.keras.layers.Layer):
    ## nn.register_parameter would be handy 
    def __init__(self, nv, nh, cd_steps):
        super(Rbm, self).__init__()
        self.W = self.add_weight(name='W', shape=(nv, nh), trainable=True)
        self.W.assign(tf.random.truncated_normal((nv, nh)) * 0.01)
        self.bv = self.add_weight(name='bias_visible', shape=(nv,)) 
        self.bh = self.add_weight(name='bias_hidden', shape=(nh,)) 
        self.cd_steps = cd_steps
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "cd_steps": self.cd_steps,
        })
        return config
        
    def sample_h(self, v):
        ph_given_v = tf.sigmoid(tf.einsum('vh,bv->bh', self.W, v) + tf.expand_dims(self.bh, axis=0))
        return self.bernoulli(ph_given_v)
      
    def sample_v(self, h):
        pv_given_h = tf.sigmoid(tf.einsum('vh,bh->bv', self.W, h) + tf.expand_dims(self.bv, axis=0))
        return self.bernoulli(pv_given_h)
    
    def bernoulli(self, p):
        # Note : No intra-batch randomness
        return tf.nn.relu(tf.sign(p - tf.random.uniform([1] + p.shape[1:])))
    
    def call(self, inputs):
        v = inputs
        # v = tf.round(inputs) # Use if input is binary # We are using intermediate layer, so skipping
        vk = tf.identity(v)
        for i in range(self.cd_steps):
            # Gibbs step
            hk = self.sample_h(vk)
            vk = self.sample_v(hk)
        # print("v", v[0][:3])
        # print("vk", vk[0][:3])
        return vk
    

class RbmLoss(tf.keras.layers.Layer):
    def __init__(self, rbm_layer):
        super(RbmLoss, self).__init__()
        
        # Cannot be serialized
        # Only used for training
        self.W = rbm_layer.W
        self.bv = rbm_layer.bv
        self.bh = rbm_layer.bh
        
    def call(self, v_in, v_out):
        # only grad for W, b
        tf.stop_gradient(v_in)
        tf.stop_gradient(v_out)
        W, bv, bh = self.W, self.bv, self.bh
        # print("W", tf.reduce_sum(W))
        loss = tf.subtract(self.energy(v_in, W, bv, bh), self.energy(v_out, W, bv, bh))
        return loss
            
    def energy(self, v, W, bv, bh):
        # NOTE : v is not normalized. If grad is not stable, then do regularization
        b_term = tf.expand_dims(tf.einsum("bx,y->b", v, bv), axis=1)
        linear_transform  = tf.einsum("bh,hx->bx", v, W) + tf.expand_dims(bh, axis=0)
        h_term = tf.expand_dims(tf.reduce_sum(tf.math.log(tf.exp(linear_transform) + 1), axis=1), axis=1)
        return tf.reduce_mean( - h_term - b_term , axis=-1)


# In[8]:


from keras.engine import data_adapter
class MultiOptimizerModel(tf.keras.Model):
    def __init__(self, DEBUG=False, **kwargs):
        super(MultiOptimizerModel, self).__init__(**kwargs)
        self.DEBUG = DEBUG
        
    def compile(self, optimizer, optimizers_and_variables_and_losses_and_name, **kwargs):
        super().compile(optimizer, **kwargs)
        self.optimizers_and_variables_and_losses_and_name = optimizers_and_variables_and_losses_and_name
        # optimizer : default optimizer
        # optimizers_and_layers_and_losses : [(optimizer, variable list, loss fn), ...] 

    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        
        loss_log = dict()
        for optimizer, variable, loss, name in self.optimizers_and_variables_and_losses_and_name:
            # Run forward pass.
            if self.DEBUG:
                with tf.GradientTape() as tape:
                    y_pred = self(x, training=True)
                    loss_value = loss(y,y_pred)
                    print("===")
                    grads = tape.gradient(loss_value, variable)
                    print(name, [(n.name, np.ravel(x.numpy())[:3]) for n, x in zip(variable, grads)])
                    print("===")

            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss_value = loss(y,y_pred)

            # Run backwards pass.
            optimizer.minimize(loss_value, variable, tape=tape)
            
            loss_log["loss_"+name] = loss_value
            
        # Use first output for validation
        output = self.compute_metrics(x, y, y_pred[0], sample_weight)
        output.update(loss_log)
        # print(loss_log, output)
        return output


# In[9]:


def modify_model(base_model, cfg, optimizer=tf.optimizers.SGD):
    if cfg["DEBUG"]:
        tf.config.run_functions_eagerly(True)
    else:
        tf.config.run_functions_eagerly(False)

    N = cfg["N"]
    INPUT_IMAGE_SIZE = cfg["INPUT_IMAGE_SIZE"]
    input_layer = tf.keras.Input((INPUT_IMAGE_SIZE,INPUT_IMAGE_SIZE,3))
    intermediate_model = tf.keras.Model(inputs=base_model.layers[-1].input, outputs=base_model.layers[-1].layers[-2].output)
    intermediate_layer = intermediate_model(tf.keras.layers.Resizing(299,299)(input_layer))

    if cfg["model"]=="default":
        output_layer = tf.keras.layers.Dense(N)(intermediate_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    elif cfg["model"]=="RBM":
        rbm_layer = Rbm(intermediate_layer.shape[-1], 1000, 3)
        rbm_loss = RbmLoss(rbm_layer)
        rbm_output = rbm_layer(intermediate_layer)
        output_layer = tf.keras.activations.sigmoid(tf.keras.layers.Dense(N)(rbm_output))
        rbm_loss_output = rbm_loss(intermediate_layer, rbm_output)
        model = MultiOptimizerModel( 
            inputs = input_layer, 
            outputs = [
                output_layer,  # (0)
                rbm_loss_output# (1)
            ],
            DEBUG = cfg["DEBUG"]
        )

    return model, optimizer

def compile_model(cfg):
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
        ],
    }
    if cfg["model"]=="default":
        kwargs["loss"] = tfa.losses.SigmoidFocalCrossEntropy()
    elif cfg["model"]=="RBM":
        all_variables = model.trainable_variables
        rbm_variables = []
        for x in model.layers:
            if x.name.split("_")[0]=="rbm":
                rbm_variables += x.trainable_variables
        rbm_variables = list({x.name:x for x in rbm_variables}.values())
        rbm_variables_name = [x.name for x in rbm_variables]
        non_rbm_variables = [x for x in all_variables if (x.name not in rbm_variables_name)]
        
        kwargs["optimizers_and_variables_and_losses_and_name"] = (
            (tf.optimizers.SGD(learning_rate=0.01), non_rbm_variables, 
             lambda y_true, y_pred : tfa.losses.sigmoid_focal_crossentropy(y_true, y_pred[0]), # (0)
             "non-rbm"),
            (tf.optimizers.SGD(learning_rate=0.01), rbm_variables, 
             lambda y_true, y_pred : y_pred[1], # (1)
             "rbm")
        )
#         kwargs["metrics"] = [
#             [],[]
#         ]
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
            with open (f"{output_path}/loss_{epoch}.json", "w") as f :
                f.write(json.dumps(logs))           
    filepath=output_path+"/weights-improvement-{epoch:02d}-{val_loss_non-rbm:.2f}.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss_non-rbm', verbose=1, save_best_only=True, mode='min')
    logdir = f"./tensorboard-logs/{datetime.isoformat(datetime.now())}"
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


# In[10]:


# cfg["DEBUG"] = True
cfg["DEBUG"] = False


# In[11]:


cfg = load_data(cfg)
cfg = prepare_dataset(cfg)
print(cfg.keys())
base_model = load_pretrained_model_01()
model, optimizer = modify_model(base_model, cfg)

model.summary()
tf.keras.utils.plot_model(model)


# In[ ]:


# tf.config.run_functions_eagerly(True)

# model.summary()
base_model.trainable=False # Freeze except rbm & dense
compile_model(cfg)
fit(model, cfg, 0, 10)

base_model.trainable=True 
compile_model(cfg)


# In[ ]:




