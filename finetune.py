from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CUDNN_LOGGING_LEVEL'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
warnings.filterwarnings('ignore')

import shutil
import argparse
from glob import glob
from PIL import Image
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import numpy as np
from model import lowlight_enhance
from utils import *

tf1.disable_eager_execution()

parser = argparse.ArgumentParser(description='Fine-tune RetinexNet')

parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--test_image', dest='test_image', default=None, help='specific test image path')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='number of epochs for fine-tuning')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, help='batch size')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=96, help='patch size')
parser.add_argument('--start_lr', dest='start_lr', type=float, default=0.0001, help='lower learning rate for fine-tuning')
parser.add_argument('--eval_every_epoch', dest='eval_every_epoch', type=int, default=10, help='eval every N epochs')

args = parser.parse_args()

def finetune_train(lowlight_enhance):
    ckpt_dir = './checkpoint'
    sample_dir = './sample'
    
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(os.path.join(ckpt_dir, 'Decom'), exist_ok=True)
    os.makedirs(os.path.join(ckpt_dir, 'Relight'), exist_ok=True)
    
    lr = [args.start_lr] * args.epoch
    lr[len(lr)//2:] = [args.start_lr * 0.1] * (args.epoch - len(lr)//2)
    
    print('[*] Loading training data...')
    train_low_data = []
    train_high_data = []
    
    train_low_data_names = glob('./data/train/input/*.*')
    train_low_data_names.sort()
    train_high_data_names = glob('./data/train/target/*.*')
    train_high_data_names.sort()
    
    print('=' * 50)
    print('FINE-TUNING RETINEXNET')
    print('=' * 50)
    print('Training pairs:', len(train_low_data))
    
    valid_pairs = []
    for idx in range(len(train_low_data_names)):
        low_im = load_images(train_low_data_names[idx])
        high_im = load_images(train_high_data_names[idx])
        if low_im.shape[2] == 3 and high_im.shape[2] == 3:
            if low_im.shape[0] >= args.patch_size and low_im.shape[1] >= args.patch_size:
                if high_im.shape[0] != low_im.shape[0] or high_im.shape[1] != low_im.shape[1]:
                    h, w = low_im.shape[:2]
                    high_im = np.array(Image.fromarray((high_im * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS)) / 255.0
                valid_pairs.append((low_im, high_im))
    
    train_low_data = [p[0] for p in valid_pairs]
    train_high_data = [p[1] for p in valid_pairs]
    print('[*] Valid pairs after filtering small images:', len(train_low_data))
    
    train_low_data = [p[0] for p in valid_pairs]
    train_high_data = [p[1] for p in valid_pairs]
    print('[*] Valid pairs after filtering small images:', len(train_low_data))
    
    eval_low_data_name = glob('./data/eval/input/*.*')
    eval_low_data = []
    for idx in range(len(eval_low_data_name)):
        eval_low_im = load_images(eval_low_data_name[idx])
        eval_low_data.append(eval_low_im)
    
    print('=' * 50)
    print('Phase 1/2: Learning decomposition')
    print('=' * 50)
    lowlight_enhance.train(
        train_low_data, train_high_data, eval_low_data,
        batch_size=args.batch_size, patch_size=args.patch_size,
        epoch=args.epoch, lr=lr, sample_dir=sample_dir,
        ckpt_dir=os.path.join(ckpt_dir, 'Decom'),
        eval_every_epoch=args.eval_every_epoch, train_phase='Decom'
    )
    
    print('=' * 50)
    print('Phase 2/2: Learning relighting')
    print('=' * 50)
    lowlight_enhance.train(
        train_low_data, train_high_data, eval_low_data,
        batch_size=args.batch_size, patch_size=args.patch_size,
        epoch=args.epoch, lr=lr, sample_dir=sample_dir,
        ckpt_dir=os.path.join(ckpt_dir, 'Relight'),
        eval_every_epoch=args.eval_every_epoch, train_phase='Relight'
    )
    
    print('[*] Fine-tuning complete!')
    print('=' * 50)
    print('TRAINING COMPLETE!')
    print('=' * 50)
    print('Checkpoints saved:')
    print('  ./checkpoint/Decom/RetinexNet_finetuned')
    print('  ./checkpoint/Relight/RetinexNet_finetuned')
    print('=' * 50)

def finetune_test(lowlight_enhance):
    save_dir = './test_results'
    os.makedirs(save_dir, exist_ok=True)
    
    if args.test_image:
        test_data = []
        test_names = []
        test_im = load_images(args.test_image)
        test_data.append(test_im)
        test_names.append(args.test_image)
    else:
        test_data_name = glob('./data/test/input/*.*')
        test_data = []
        test_names = []
        for idx in range(len(test_data_name)):
            test_im = load_images(test_data_name[idx])
            test_data.append(test_im)
            test_names.append(test_data_name[idx])
    
    tf1.global_variables_initializer().run()
    
    print('[*] Loading fine-tuned model...')
    load_model_status_Decom, _ = lowlight_enhance.load(lowlight_enhance.saver_Decom, './checkpoint/Decom')
    load_model_status_Relight, _ = lowlight_enhance.load(lowlight_enhance.saver_Relight, './checkpoint/Relight')
    
    if load_model_status_Decom and load_model_status_Relight:
        print('[*] Fine-tuned model loaded!')
        
        for idx in range(len(test_data)):
            [_, name] = os.path.split(test_names[idx])
            suffix = name[name.find('.') + 1:]
            name = name[:name.find('.')]
            
            input_test = np.expand_dims(test_data[idx], axis=0)
            [R_low, I_low, I_delta, S] = lowlight_enhance.sess.run(
                [lowlight_enhance.output_R_low, lowlight_enhance.output_I_low, 
                 lowlight_enhance.output_I_delta, lowlight_enhance.output_S],
                feed_dict={lowlight_enhance.input_low: input_test}
            )
            
            save_images(os.path.join(save_dir, name + "_S." + suffix), S)
            print('[*] Saved:', name + "_S." + suffix)
        
        print('[*] Results saved in', save_dir)
    else:
        print('[!] Failed to load fine-tuned model')

def main(_):
    gpu_options = tf1.GPUOptions(per_process_gpu_memory_fraction=0.5)
    with tf1.Session(config=tf1.ConfigProto(gpu_options=gpu_options)) as sess:
        model = lowlight_enhance(sess)
        
        if args.phase == 'train':
            finetune_train(model)
        elif args.phase == 'test':
            finetune_test(model)
        else:
            print('[!] Unknown phase')

if __name__ == '__main__':
    tf1.app.run()