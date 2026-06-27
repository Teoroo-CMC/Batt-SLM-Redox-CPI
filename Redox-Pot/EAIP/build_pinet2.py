import os
from pathlib import Path

EAIP_DIR = Path("your_own_path/Batt-SLM-Redox-CPI/Redox-Pot/EAIP/")

# create predictor directory
if not os.path.exists(EAIP_DIR / "Models"):
    os.mkdir(EAIP_DIR / "Models")

# build PiNet2-P3 Model
def build_pinet2_model(target_label, random_seed):

    import yaml
    import warnings
    import tensorflow as tf
    from pinn import get_model, get_network
    from pinn.utils import init_params
    from pinn.io import load_hdf5, write_tfrecord, load_tfrecord, sparse_batch
    from tensorflow.python.lib.io.file_io import FileIO
    from tempfile import mkdtemp, mkstemp

    # important parameters
    depth = 5
    cutoff = 4.5
    gauss_basis = 15 # 15 is the best
    model_size = 64
    learning_rate = 0.0001
    learning_rate_decay = 0.98
    if target_label == "dipole":
        learning_rate_decay = 0.994
    training_step = 3000000
    eval_step = 100
    batch_size = 10
    split_training_ratio = 9

    # other parameters
    nlog_every = 1000
    nckpt_every = 10000
    nmax_ckpts = 1
    shuffle_buffer = 1000
    is_shuffle = True
    is_preprocess = True
    is_using_cache = True
    is_early_stop = False

    # set the GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # turn off the specific warning of tensorflow
    # comment this block, if you want tensorflow print something via logging
    # index_warning = 'Converting sparse IndexedSlices'
    # warnings.filterwarnings('ignore', index_warning)
    # tf.get_logger().setLevel('ERROR')

    label_to_folder = {"homo":"HOMO", "lumo":"LUMO", "ea":"EA", "ip":"IP", "dipole":"Dipole"}
    lable_predictor_dir = EAIP_DIR / "Models" / label_to_folder[target_label]
    if not os.path.exists(lable_predictor_dir):
        os.mkdir(lable_predictor_dir)
    print(lable_predictor_dir)

    # split training and test sets
    key_in_pinn = "e_data"
    if target_label == "dipole":
        key_in_pinn = "d_data"
    dataset = load_hdf5(str(EAIP_DIR / "Batt-P30K.h5"), label_map={key_in_pinn: target_label}, splits={'train':split_training_ratio,
                        'vali':10-split_training_ratio}, shuffle=is_shuffle, seed=random_seed)
    write_tfrecord(str(lable_predictor_dir) + '-train-%s-%d.yml'%(target_label, random_seed), dataset['train'])
    write_tfrecord(str(lable_predictor_dir) + '-vali-%s-%d.yml'%(target_label, random_seed), dataset['vali'])

    # get model parameters
    params = {}

    # define model
    if target_label == "dipole":
        params["model"] = { "name":  "AD_dipole_model",
                            "params":
                                 {  "d_scale": 2.5412,
                                    "d_unit": 2.5412,
                                 }
                          }
    else:
        params["model"] = { "name": "potential_model",
                            "params":
                                 {  "use_force": False,
                                    "e_loss_multiplier": 1.0,
                                    "f_loss_multiplier": 10.0,
                                    "e_scale": 1.0,
                                    "e_unit": 1.0,
                                    "use_e_per_atom": False,
                                    "log_e_per_atom": True,
                                 }
                           }

    # model para
    params["network"] = {   "name": "PiNet2",
                            "params":
                                {  "depth": depth,
                                    "rc": cutoff,
                                    "n_basis": gauss_basis,
                                    "basis_type": "gaussian",
                                    "pi_nodes": [model_size],
                                    "pp_nodes": [model_size]*4,
                                    "ii_nodes": [model_size]*4,
                                    "out_nodes": [model_size],
                                    "rank": 3,
                                    "weighted": False
                                }
                            }
    if target_label == "dipole":
        params["network"]["params"]["atom_types"] = [1, 6, 7, 8, 9, 15, 16, 17]
        if "AD" in params["model"]["name"]:
            params["network"]["params"]["out_extra"] = {"p1":1, "p3":1}

    # adam
    params["optimizer"] = {     "class_name": "Adam",
                                "config":
                                    {
                                        "global_clipnorm": 0.01,
                                        "learning_rate":
                                            {
                                                "class_name": "ExponentialDecay",
                                                "config":
                                                {
                                                    "decay_rate": learning_rate_decay,
                                                    "decay_steps": 100000,
                                                    "initial_learning_rate": learning_rate,
                                                }
                                            }
                                    }
                            }
    if target_label == "dipole":
        params["optimizer"]["config"]["learning_rate"]["config"]["decay_steps"] = 10000

    import re
    def shorten(x):
        s = f"{float(x):.5E}"
        s = re.sub(r'\.?0*E', 'E', s)
        s = re.sub(r'E\+0*', 'E', s)
        return s

    model_name = "PiNet2-%s-B%d-%s-%d"%(label_to_folder[target_label], batch_size, shorten(training_step), random_seed)
    model_full_path = lable_predictor_dir / model_name
    if os.path.exists(model_full_path):
        print("Note: continue training!!")

    params['model_dir'] = str(model_full_path)

    # initial e_dress is necessary, even running in a fine-tuning proceess
    ds = load_tfrecord(str(lable_predictor_dir) + '-train-%s-%d.yml'%(target_label, random_seed))
    init_params(params, ds)

    scratch_dir = None
    if scratch_dir is not None:
        scratch_dir = mkdtemp(prefix='pinn', dir=scratch_dir)
    def _dataset_fn(fname):
        dataset = load_tfrecord(fname)
        if batch_size is not None:
            dataset = dataset.apply(sparse_batch(batch_size))
        if is_preprocess:
            def pre_fn(tensors):
                with tf.name_scope("PRE") as scope:
                    network = get_network(params['network'])
                    tensors = network.preprocess(tensors)
                return tensors
            dataset = dataset.map(pre_fn)
        if is_using_cache:
            if scratch_dir is not None:
                cache_dir = mkstemp(dir=scratch_dir)
            else:
                cache_dir = ''
            dataset = dataset.cache(cache_dir)
        return dataset

    train_fn = lambda: _dataset_fn(str(lable_predictor_dir) + '-train-%s-%d.yml'%(target_label, random_seed)).repeat().shuffle(shuffle_buffer)
    eval_fn = lambda: _dataset_fn(str(lable_predictor_dir) + '-vali-%s-%d.yml'%(target_label, random_seed))
    config = tf.estimator.RunConfig(keep_checkpoint_max=nmax_ckpts,
                                    log_step_count_steps=nlog_every,
                                    save_summary_steps=nlog_every,
                                    save_checkpoints_steps=nckpt_every)

    model = get_model(params, config=config)

    if is_early_stop:
        early_stop = "loss:1000"
        stops = {s.split(':')[0]: float(s.split(':')[1])
                 for s in early_stop.split(',')}
        hooks = [tf.estimator.experimental.stop_if_no_decrease_hook(
            model, k, v) for k,v in stops.items()]
    else:
        hooks=None

    # tensorflow set the mode=tf.estimator.ModeKeys.TRAIN atomaticlly
    train_spec = tf.estimator.TrainSpec(input_fn=train_fn, max_steps=training_step, hooks=hooks)

    # tensorflow set the mode=tf.estimator.ModeKeys.EVAL atomaticlly
    eval_spec  = tf.estimator.EvalSpec(input_fn=eval_fn, steps=eval_step)
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

#
if __name__ == '__main__':

    # run it by slurm
    target_label = "ea" # "ea", "ip"
    random_seed = 6   # 6, 7, 8
    build_pinet2_model(target_label, random_seed)
