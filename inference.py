from itertools import product
import os

import tensorflow as tf 
import numpy as np
import numpy.ma as ma
import json
import cv2
from PIL import Image

from Data_utils import preprocessing, weights_utils
import Nets 
from Sampler import sampler_factory
from Losses import loss_factory


class MADNet(object):
    def __init__(self, model_name='MADNet',
                weight_path=None,
                learning_rate=0.0001,
                block_config_path='block_config/MadNet_full.json',
                image_shape=[480,640],
                crop_shape=[None,None],
                SSIMTh=0.5,
                mode='MAD'):
        self._model_name = model_name
        self._weight_path = weight_path
        self._learning_rate = learning_rate
        self._block_config_path = block_config_path
        self._image_shape = image_shape
        self._crop_shape = crop_shape
        self._SSIMTh = SSIMTh
        self._mode = mode 
        self._ready = self._setup_graph()
        self._ready &= self._initialize_model()
        self.first = True

    def _load_block_config(self):
        #load json file config
        with open(self._block_config_path) as json_data:
            self._train_config = json.load(json_data)

    def _build_input_ops(self):
        #input placeholder ops
        self._left_placeholder = tf.placeholder(tf.float32,shape=[1,None, None,3], name='left_input')
        self._right_placeholder = tf.placeholder(tf.float32,shape=[1,None, None,3], name='right_input')

        self._left_input = self._left_placeholder
        self._right_input = self._right_placeholder
        
        if self._image_shape[0] is not None:
            self._left_input = preprocessing.rescale_image(self._left_input, self._image_shape)
            self._right_input = preprocessing.rescale_image(self._right_input, self._image_shape)
        
        if self._crop_shape[0] is not None:
            self._left_input = tf.image.resize_image_with_crop_or_pad(self._left_input, self._crop_shape[0], self._crop_shape[1])
            self._right_input = tf.image.resize_image_with_crop_or_pad(self._right_input, self._crop_shape[0], self._crop_shape[1])

    def _build_network(self):
        #network model
        with tf.variable_scope('model'):
            net_args = {}
            net_args['left_img'] = self._left_input
            net_args['right_img'] = self._right_input
            net_args['split_layers'] = [None]
            net_args['sequence'] = True
            net_args['train_portion'] = 'BEGIN'
            net_args['bulkhead'] = True if self._mode=='MAD' else False
            self._net = Nets.get_stereo_net(self._model_name, net_args)
            self._predictions = self._net.get_disparities()
            self._full_res_disp = self._predictions[-1]

            self._inputs = {
                'left':self._left_input,
                'right':self._right_input,
                'target':tf.zeros([1,self._image_shape[0],self._image_shape[1],1],dtype=tf.float32)
            }

            #full resolution loss between warped right image and original left image
            self._loss =  loss_factory.get_reprojection_loss('mean_SSIM_l1',reduced=True)(self._predictions,self._inputs)
    
    def _MAD_adaptation_ops(self):
        #build train ops for separate portions of the network
        self._load_block_config()

        #keep all predictions except full res
        predictions = self._predictions[:-1] 
        
        inputs_modules = self._inputs
        
        assert(len(predictions)==len(self._train_config))
        for counter,p in enumerate(predictions):
            print('Build train ops for disparity {}'.format(counter))
                    
            #rescale predictions to proper resolution
            multiplier = tf.cast(tf.shape(self._left_input)[1]//tf.shape(p)[1],tf.float32)
            p = preprocessing.resize_to_prediction(p,inputs_modules['left'])*multiplier

            #compute reprojection error
            with tf.variable_scope('reprojection_'+str(counter)):
                reconstruction_loss = loss_factory.get_reprojection_loss('mean_SSIM_l1',reduced=True)([p],inputs_modules)

            #build train op
            layer_to_train = self._train_config[counter]
            print('Going to train on {}'.format(layer_to_train))
            var_accumulator=[]
            for name in layer_to_train:
                var_accumulator+=self._net.get_variables(name)
            print('Number of variable to train: {}'.format(len(var_accumulator)))
                
            #add new training op
            self._train_ops.append(self._trainer.minimize(reconstruction_loss,var_list=var_accumulator))

            print('Done')
            print('='*50)
        
        #create Sampler to fetch portions to train
        self._sampler = sampler_factory.get_sampler('PROBABILITY',1,0)
    
    def _Full_adaptation_ops(self):
        self._train_ops.append(self._trainer.minimize(self._loss))
        self._sampler = sampler_factory.get_sampler('FIXED',1,0)
    
    def _no_adaptation_ops(self):
        #mock ops that don't do anything
        self._train_ops.append(tf.no_op())
        self._sampler = sampler_factory.get_sampler('FIXED',1,0)

    def _build_adaptation_ops(self):
        """
        Populate self._train_ops
        """
        #self._trainer = tf.train.MomentumOptimizer(self._learning_rate,0.9)
        self._trainer = tf.train.AdamOptimizer(self._learning_rate)
        self._train_ops = []
        if self._mode == 'MAD':
            self._MAD_adaptation_ops()
        elif self._mode == 'FULL':
            self._Full_adaptation_ops()
        elif self._mode == 'NONE':
            self._no_adaptation_ops()

    def _setup_graph(self):
        """
        Build tensorflow graph and ops
        """
        self._build_input_ops()

        self._build_network()

        self._build_adaptation_ops()

        return True
    
    def _initialize_model(self):
        """
        Create tensorflow session and initialize the network
        """
        #session
        gpu_options = tf.GPUOptions(allow_growth=True)
        self._session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        #variable initialization
        initializers = [tf.global_variables_initializer(),tf.local_variables_initializer()]
        self._session.run(initializers)

        #restore disparity inference weights and populate self._restore_op
        if self._weight_path is not None:
            var_to_restore = weights_utils.get_var_to_restore_list(self._weight_path, [])
            self._restorer = tf.train.Saver(var_list=var_to_restore)
            self._restore_op = lambda: self._restorer.restore(self._session, self._weight_path)
            self._restore_op()
            print('Disparity Net Restored?: {}, number of restored variables: {}'.format(True,len(var_to_restore)))
        else:
            self._restore_op = lambda: self._session.run(initializers)

        #operation to select the different train ops
        num_actions=len(self._train_ops)
        self._sample_distribution=np.zeros(shape=[num_actions])
        self._temp_score = np.zeros(shape=[num_actions])
        self._loss_t_2 = 0
        self._loss_t_1 = 0
        self._expected_loss = 0
        self._last_trained_blocks = 0

        print('Network Ready')

        return True

    def inference(self, left_frame, right_frame):
        #Fetch portion of network to train
        #softmax
        exp = np.exp(self._sample_distribution)
        distribution = exp/np.sum(exp,axis=0)
        train_op_id = self._sampler.sample(distribution)[0]
        selected_train_op = self._train_ops[train_op_id]

        #build list of tensorflow operations that needs to be executed + feed dict
        tf_fetches = [self._loss, selected_train_op, self._full_res_disp, self._left_input, self._right_input]
        fd = {
            self._left_placeholder: left_frame,
            self._right_placeholder: right_frame
        }

        #run network
        full_ssim, _, disp_prediction, lefty, righty = self._session.run(tf_fetches, feed_dict=fd)

        if self._mode == 'MAD':
            #update sampling probabilities
            if self.first:
                self._loss_t_2 = full_ssim
                self._loss_t_1 = full_ssim
                self.first = False

            self._expected_loss = 2*self._loss_t_1-self._loss_t_2	
            gain_loss=self._expected_loss-full_ssim
            self._sample_distribution = 0.99*self._sample_distribution
            self._sample_distribution[self._last_trained_blocks] += 0.01*gain_loss

            self._last_trained_blocks = train_op_id
            self._loss_t_2 = self._loss_t_1
            self._loss_t_1 = full_ssim
        
        if full_ssim > self._SSIMTh:
            print('Resetting Network...')
            self._restore_op()
        
        return disp_prediction
    
    def save(self, save_path):
        saver = tf.train.Saver()
        saver.save(self._session, save_path)

    def close(self):
        self._session.close()


def build_input_images(image_folder):
    images = [i for i in os.listdir(image_folder) if i.endswith(("jpg", "png"))]
    left_images = [os.path.join(image_folder, i) for i in images if i.__contains__("left")]
    right_images = [os.path.join(image_folder, i) for i in images if i.__contains__("right")]

    left_images.sort()
    right_images.sort()

    assert len(left_images) == len(right_images)
    return left_images, right_images


def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")
    return np.array(image)


def load_roi_file(roi_file_path):
    result = {}
    with open(roi_file_path, "r") as f:
        result = {i.split()[0]: [int(j) if j.isdigit() else float(j) for j in i.split()[1:]] for i in f.readlines()}
    
    return result


def load_stereo_coefficients(path):
    """ Loads stereo matrix coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    K1 = cv_file.getNode("K1").mat()
    D1 = cv_file.getNode("D1").mat()
    K2 = cv_file.getNode("K2").mat()
    D2 = cv_file.getNode("D2").mat()
    R = cv_file.getNode("R").mat()
    T = cv_file.getNode("T").mat()
    E = cv_file.getNode("E").mat()
    F = cv_file.getNode("F").mat()
    R1 = cv_file.getNode("R1").mat()
    R2 = cv_file.getNode("R2").mat()
    P1 = cv_file.getNode("P1").mat()
    P2 = cv_file.getNode("P2").mat()
    Q = cv_file.getNode("Q").mat()

    cv_file.release()
    return [K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q]


if __name__ == "__main__":
    # model_no_adaptation = MADNet(weight_path="checkpoints/MADNet/kitti/weights.ckpt", image_shape=[480, 640], mode='NONE')
    model_outdoor = MADNet(weight_path="results/outdoor_p30_accuracy_pretraining/weights", image_shape=[480, 640], mode='NONE')
    # model_outdoor = MADNet(weight_path="results/outdoor_accuracy_pretraining/weights", image_shape=[480, 640], mode='NONE')
    # model_no_adaptation = MADNet(weight_path="checkpoints/MADNet/kitti/weights.ckpt", image_shape=[640, 480], mode='NONE')
    # model_indoor = MADNet(weight_path="results/indoor_accuracy_pretraining/weights", image_shape=[640, 480], mode='NONE')
    # model_outdoor = MADNet(weight_path="results/outdoor_accuracy_pretraining/weights", image_shape=[640, 480], mode='NONE')
    # phone_type = "mate40pro"
    phone_type = "p30pro"
    stereo_calibration_file = f"../AnyNet/calib_result/{phone_type}.yml"
    
    root_image_folder = "images/p30_new"
    modes = ["processed_static", "processed_slow", "processed_fast"]
    # depth_folders = ["0.5", "1"]
    depth_folders = ["3", "5"]
    min_depth = -10
    max_depth = 20
    if 'model_outdoor' in globals():
        model = globals()['model_outdoor'] 
        model_type = 'mad_adaptation'
    elif 'model_indoor' in globals():
        model = globals()['model_indoor']
        model_type = 'mad_adaptation'
    elif 'model_no_adaptation' in globals():
        model = globals()['model_no_adaptation']
        model_type = 'no_adaptation'
    else:
        raise ValueError("wrong model")

    root_output_folder = f"output3/{model_type}/{phone_type}"

    Q = load_stereo_coefficients(stereo_calibration_file)[-1]

    for depth_folder, mode in product(depth_folders, modes):
        result_save_path = os.path.join(root_output_folder, f"{mode}_{depth_folder}.csv")
        result = []
        gt_to_error_rate = {}

        output_folder = os.path.join(root_output_folder, depth_folder, mode)
        raw_disp_output_folder = os.path.join(root_output_folder, depth_folder, mode.replace("processed", "disparity"))
        image_folder = os.path.join(root_image_folder, depth_folder, mode)
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(raw_disp_output_folder, exist_ok=True)
        left_image_paths, right_image_paths = build_input_images(image_folder)

        for _, (left_image_path, right_image_path) in enumerate(zip(left_image_paths, right_image_paths)):
            imgL = load_image(left_image_path)[np.newaxis, ...]
            imgR = load_image(right_image_path)[np.newaxis, ...]

            outputs = model.inference(imgL, imgR)
            disp = np.squeeze(outputs[-1])

            disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            disp_vis = 255 - disp_vis
            disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)

            points_3d = cv2.reprojectImageTo3D(disp, Q)
            depth_map = points_3d[:, :, -1]
            depth_map = np.clip(depth_map, min_depth, max_depth)
            depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_map = 255 - depth_map
            depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_TURBO)
            
            left_image = cv2.cvtColor(imgL[0], cv2.COLOR_RGB2BGR)
            cv2.putText(left_image, "origin", (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(disp_vis, "dispraity", (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(depth_map, "depth", (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(output_folder, os.path.basename(left_image_path)), np.hstack([left_image, disp_vis, depth_map]))

            np.save(os.path.join(raw_disp_output_folder, os.path.basename(left_image_path).replace(".png", "_madnet.npy")), disp)

