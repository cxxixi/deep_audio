from torch.utils.data import Dataset, DataLoader
from collections import Counter
import utils as U
from utils import *
import pandas



class audio_esc50(Dataset):
    
    def __init__(self,train, args, indices):
        
        self.train = train
        self.args = args
        self.preprocess_funcs = self.preprocess_setup()
#         print(indices)
#         sound = wavio.read(wav_file).data.T[0]
#         start = sound.nonzero()[0].min()
#         end = sound.nonzero()[0].max()
#         sound = sound[start: end + 1]
    
#         self.audio_files = os.listdir('./datasets/ESC-50-master/audio/')
    
        self.audio_files = './datasets/ESC-50-master/audio/'
        
        _ = pd.read_csv('./datasets/ESC-50-master/meta/esc50.csv')
        dataset = pd.DataFrame()
        dataset['fname'] = _['filename']
        dataset['label'] = _['target']
        dataset = dataset.iloc[indices]

        t1 = time.time()
        mfcc_feature = dataset['fname'].apply(self.get_mfcc, path=self.audio_files)
        t2 = time.time()
        print('Generating mfcc features done , execution time:{}s'.format(t2 - t1))
        spectro_feature = dataset['fname'].apply(self.spectral_features, path=self.audio_files)
        t3 = time.time()
        print('Generatings spetral features done , execution time:{}s'.format(t3 - t2))
        
        if not self.train:
#             print(mfcc_feature.shape)
            mfcc_feature = np.resize(mfcc_feature, (mfcc_feature.shape[0]*self.args.nCrops, mfcc_feature.shape[1]//self.args.nCrops))
    #         mfcc_feature = mfcc_feature.reshape(mfcc_feature.shape[0]*10, -1)
#             print(mfcc_feature.shape)
#             print(spectro_feature.shape)
            spectro_feature = np.resize(spectro_feature, (spectro_feature.shape[0]*self.args.nCrops, spectro_feature.shape[1]//self.args.nCrops))
#             print(spectro_feature.shape)
            _features = np.concatenate([mfcc_feature,spectro_feature], axis=1)
            _features = np.resize(_features, (_features.shape[0]//self.args.nCrops, _features.shape[1]*self.args.nCrops))
#             print(_features.shape)

            _features = pd.DataFrame(_features)
    #         self.dataset = pd.concat([dataset,mfcc_feature], axis=1)
            self.dataset = pd.concat([dataset,_features], axis=1)
            print(_features.shape)
        else:
            self.dataset = pd.concat([dataset,mfcc_feature, spectro_feature], axis=1)
        
        self.dataset.to_csv("./esc50.csv")

#         self.dataset = pd.read_csv("./esc50.csv")
        self.data_list = np.asarray(self.dataset.drop(["label",'fname'], axis=1))
        self.label_list = np.asarray(self.dataset["label"])
        
    
    def get_mfcc(self, name, path):
        
        sr = 44100
        b, _ = librosa.core.load(path + name, sr = sr)
#         b = self.preprocess(b)
        if self.train:
            gmm = librosa.feature.mfcc(b, sr = sr, n_mfcc=80, lifter=140)
            return pd.Series(np.hstack((np.mean(gmm, axis=1), np.std(gmm, axis=1), skew(gmm, axis = 1), np.median(gmm, axis = 1))))
        else:
            features = []
            for i in range(len(b)):
                gmm = librosa.feature.mfcc(b[i], sr = sr, n_mfcc=80, lifter=140)
                features.append(pd.Series(np.hstack((np.mean(gmm, axis=1), np.std(gmm, axis=1), skew(gmm, axis = 1), np.median(gmm, axis = 1)))))
#             print(np.concatenate(features).shape)
            return pd.Series(np.concatenate(features))
                
    def window(self, S, xsize=3, ysize=6):
        S_dB = librosa.power_to_db(S)
        S_dB = S_dB.astype(np.float32)

        x, y = S_dB.shape
        size_x = xsize
        size_y = ysize
        n_frame_x = x//size_x
        n_frame_y = y//size_y
        features= []
        count = 0
        for i in range(n_frame_x-1):
            for j in range(n_frame_y-1):
                count +=1
                window = S_dB[i*size_x:(i+1)*size_x, j*size_y:(j+1)*size_y]
                _max = np.max(window)
                features.append(_max)
         
        return features


    def spectral_features(self, name, path):
        
        SR = 44100
        (sig, rate) = librosa.load(path + name, sr=None, mono=True,  dtype=np.float32)
#         sig = self.preprocess(sig)
        if self.train:
            S = librosa.feature.melspectrogram(y=sig, sr=SR)
            features = self.window(S)
            return pd.Series(features)
        else:
            features = []
            for i in range(len(sig)):
                S = librosa.feature.melspectrogram(y=sig[i], sr=SR)
                feature = self.window(S)
                features.append(feature)
        
#         print(pd.concat(features).shape)
            return pd.Series(np.concatenate(features))   
            
        


 
    def __len__(self):
        return len(self.data_list)
    
    
    def preprocess(self, sound):
        for f in self.preprocess_funcs:
            sound = f(sound)

        return sound

    def preprocess_setup(self):
#         if self.train:
#             funcs = []
# #             if self.args.strongAugment:
# #                 funcs += [U.random_scale(1.25)]

#             funcs += [U.padding(self.args.inputLength // 2),
#                       U.random_crop(self.args.inputLength),
#                       U.normalize(32768.0),
#                       ]

#         else:
#             funcs = [U.padding(self.args.inputLength // 2),
#                      U.normalize(32768.0),
#                      U.multi_crop(self.args.inputLength, self.args.nCrops),
#                      ]
            
        if self.train:
            funcs = []
#             if self.args.strongAugment:
#                 funcs += [U.random_scale(1.25)]

            funcs += [
                      U.random_crop(self.args.inputLength),
                      U.normalize(32768.0),
                      ]

        else:
            funcs = [
                     U.normalize(32768.0),
                     U.multi_crop(self.args.inputLength, self.args.nCrops),
                     ]
        return funcs

    def __getitem__(self, index):
        
        data = self.data_list[index]
        label= self.label_list[index] 
        return data, label

if __name__ == "__main__": 
    
    validation_split = .01
    indices = list(range(2000))
    split = int(np.floor(validation_split * 2000))
    if True:
        np.random.seed(42)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    data = audio_esc50(train=False,args=args, indices=val_indices)
#     print(data.__getitem__(1))
