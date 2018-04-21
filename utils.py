import os   
import librosa
import numpy as np
from tqdm import tqdm
import pickle as pkl
import essentia

print essentia.__version__

# import pdb; pdb.set_trace()

def find_disfluency(start_time, start_ind, end_ind,ctm_file, model_output_file):

    with open(model_output_file) as out,open(ctm_file) as file:
        model_output_file = out.read()
        ctm_file = file.read()

    model_output = [word.split('/') for word in model_output_file.strip('\n').split(' ')]
    ctm = [word.strip().split(' ') for word in ctm_file.strip('\n').split('\n')]
    
    print("indexing mein gadbad hai mostly")
    model_output = model_output[1:]
    ctm = [word for word in ctm if word[4][0] !='[']
    
    disfluencies = []    

    ind = start_ind
    dummy_end_time = None
    labels = {'B-D': 'MD','B-I':'MF','S-D':'SD','S-I':'SI'}
    while ind < end_ind:
        try:
            if model_output[ind][2] in ['B-D','B-I', 'S-D','S-I']: 
                disfluencies.append([float(ctm[ind][2])*3000 - start_time,dummy_end_time,labels[model_output[ind][2]]])

                if model_output[ind][2] in ['B-D','B-I']:
                    while model_output[ind][2] not in ['E-D','E-I']: # assuming there's always an end tag for all disfluencies
                        ind+=1
                disfluencies[-1][1] = round(float(ctm[ind][2]) + float(ctm[ind][3]),2)*3000 - start_time # end_time = start_time of last word + duration
        except:
            print(ind)

        ind+=1
    # print( disfluencies)
    return disfluencies

            


# find_disfluency(1,500,'./transcript.ctm','./out.txt')


def ctm_index_from_time(time,ctm, end_ind = False):
    
    for i in range(len(ctm)):
        if round(float(ctm[i][2])*3000,2) >= time:
            if i > 0:
                if end_ind:
                    return i-1
            
            return i
            
    return len(ctm)-1

def umm_tracker(start_time, start_ind, end_ind, ctm):
    # with open(ctm_file) as file:
    #     ctm_file = file.read()

    # ctm = [word.strip().split(' ') for word in ctm_file.strip('\n').split('\n')]
    # ctm = [word for word in ctm if word[4][0] !='[']
    
    disfluencies = []  

    ind = start_ind
    while ind <= end_ind:
        if ctm[ind][4] in ['uh','uhh','um','umm']: 
            d = [round(float(ctm[ind][2])*3000 - start_time,2), round((float(ctm[ind][2]) + float(ctm[ind][3]))*3000,2) - start_time,1]
            disfluencies.append(d)
            
        ind+=1
    return disfluencies


def parse_data(data_dir):
    all_files = os.listdir(data_dir) 
    features = np.zeros((len(all_files)/2,193)) # feature vector length
    labels = np.zeros(len(all_files)/2)
    ind = 0
    for file in tqdm(all_files):
        if file.endswith(".wav"):
            # mfccs, chroma, mel, contrast,tonnetz = extract_feature(os.path.join(data_dir, file))
            # features[ind] = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            data_file_name = file[:-4]
            with open(os.path.join(data_dir, data_file_name + "_label.txt")) as l:
                label = l.read()
            if label:
                labels[ind] = 1 #discrepancy
            ind+=1
            # else:
            #     labels = np.append(labels,1) # normal clip
            # label

    return features,labels

def get_balanced_data(data_dir,pkl_dump_name):
    positive_samples = 1822
    features = np.zeros((4000,193)) # feature vector length
    labels = np.zeros(4000)
    max_neg = 4000 - positive_samples
    ind = 0
    neg_count = 0
    for file in tqdm(os.listdir(data_dir)):
        if file.endswith(".wav"):
            data_file_name = file[:-4]
            with open(os.path.join(data_dir, data_file_name + "_label.txt")) as l:
                label = l.read()
            if label:
                labels[ind] = 1 #discrepancy
            elif neg_count >=max_neg:
                continue
            else:
                neg_count+=1
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(os.path.join(data_dir, file))
            features[ind] = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            ind+=1
            # else:
            #     labels = np.append(labels,1) # normal clip
            # label

    with open('./data/'+pkl_dump_name+'.pkl','wb') as f:
        pkl.dump((features,labels),f,protocol=2)
    return features,labels

def get_umm_balanced_regression_data(data_dir,pkl_dump_name):
    features = np.zeros((1060,193)) # feature vector length
    labels = np.zeros((1060,2))
    ind = 0
    neg_count = 0
    for file in tqdm(os.listdir(data_dir)):
        if file.endswith(".wav"):
            data_file_name = file[:-4]
            with open(os.path.join(data_dir, data_file_name + "_label.txt")) as l:
                label = l.read()
            if label:
                label = [float(elem) for elem in list(label.strip().split(" "))]
                print('debug')
                labels[ind] = label[:2] #discrepancy start and end time
            elif neg_count >550:
                continue
            else:
                neg_count+=1
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(os.path.join(data_dir, file))
            features[ind] = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            ind+=1

    with open('./data/'+pkl_dump_name+'.pkl','w') as f:
        pkl.dump((features,labels),f,protocol=2)
    return features,labels  


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz


# data = parse_data("./TrainingDataUmm")
# get_balanced_data("./TrainingDataUmm",'umm_balanced')

# get_umm_balanced_regression_data("./TrainingDataUmm",'umm_balanced_regression')
# print("yolo")
