import sys
import pandas as pd
import os
import librosa
import numpy as np
import librosa.display
import time 
import subprocess
# import utils 

def generate_esc50():
    
    esc50_dir = './'
    subprocess.call('wget -P {} https://github.com/karoldvl/ESC-50/archive/master.zip'.format(
    esc50_dir), shell=True)
    subprocess.call('unzip -d {} {}'.format(
        esc50_dir, os.path.join(esc50_dir, 'master.zip')), shell=True)
    os.remove(os.path.join(esc50_dir, 'master.zip'))

    audio_dir = './ESC-50-master/audio/'
    SR = 16000

    t1 = time.time()
    for fileList in os.walk(audio_dir):
        count = 0
        for fname in fileList[2]:
            count += 1
            name_array = fname.split('.')
            name = name_array[0]
            name_array = name_array[0].split('-')
            curr_class = name_array[-1]
            if(name != '' and curr_class != ''):
                fpath = audio_dir + fname
                (sig, rate) = librosa.load(fpath, sr=None, mono=True,  dtype=np.float32)
                S = librosa.feature.melspectrogram(y=sig, sr=SR)
                S_dB = librosa.power_to_db(S)
                S_dB = S_dB.astype(np.float32)

                save_dir = './ESC-50-master/spectrograms/class' + str(curr_class) + '/'

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                np.save(save_dir + name + '.npy',S_dB)
    t2 = time.time()
    print("Total Spectrograms Generation Time: {}s".format(t2-t1))

def generate_fsd():
    
    fsd_path = "./fsd"
    os.mkdir(fsd_path)

    # Download freesound from kaggle
    subprocess.call('kaggle competitions download -c freesound-audio-tagging -p {}'.format(
        fsd_path), shell=True)
    subprocess.call('unzip -d {} {}'.format(
        fsd_path, os.path.join(fsd_path, 'freesound-audio-tagging.zip')), shell=True)
    os.remove(os.path.join(fsd_path, 'freesound-audio-tagging.zip'))
    
    audio_dir = './ESC-50-master/audio/'
    SR = 16000
    
    t1 = time.time()
    for fileList in os.walk(audio_dir):
        count = 0
        for fname in fileList[2]:
            count += 1
            name_array = fname.split('.')
            name = name_array[0]
            name_array = name_array[0].split('-')
            curr_class = name_array[-1]
            if(name != '' and curr_class != ''):
                fpath = audio_dir + fname
                (sig, rate) = librosa.load(fpath, sr=None, mono=True,  dtype=np.float32)
                S = librosa.feature.melspectrogram(y=sig, sr=SR)
                S_dB = librosa.power_to_db(S)
                S_dB = S_dB.astype(np.float32)

                save_dir = './ESC-50-master/spectrograms/class' + str(curr_class) + '/'

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                np.save(save_dir + name + '.npy',S_dB)
    t2 = time.time()
    print("Total Spectrograms Generation Time: {}s".format(t2-t1))
    

if __name__ == "__main__":

#     generate_esc50()
#     generate_fsd()
    
#     elif args.dataset == "fsd":
#         pass

