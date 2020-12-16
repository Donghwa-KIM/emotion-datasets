# main 
import os
import h5py

# audio package
import librosa
import librosa.display
import IPython.display as ipd # ipd.Audio(arr, rate=seg_audio.frame_rate) 
from pydub import AudioSegment # sudo apt-get install ffmpeg

# basic
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict, OrderedDict
import re

import logging


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)



class CmuMoseiDataset:
    def __init__(self, meta_root, comb_root, audio_root, label_path):
        self.meta_root = meta_root
        self.comb_root = comb_root
        self.audio_root = audio_root
        self.label_path = label_path
    # load label file
    def get_label_df(self):
        h5handle = h5py.File(self.label_path,'r')
        label_data = dict(h5handle[list(h5handle.keys())[0]]["data"])
        metadata = dict(h5handle[list(h5handle.keys())[0]]["metadata"])

        # init dataframe
        colnames = ['wav_id','start','end'] + eval(metadata['dimension names'][()][0]) 


        label_dataset = []
        for ix in label_data:
            N = label_data[ix]['features'].shape[0]
            label_per_wav = np.concatenate([label_data[ix]['intervals'][()],
                                            label_data[ix]['features'][()]], axis=1).tolist()
            label_per_wav = [[ix]+row  for row in label_per_wav]
            label_dataset.extend(label_per_wav)

        label_df = pd.DataFrame(label_dataset, columns= colnames)
        return label_df

    def add_data_ix(self, data_ix ,split_name):
        for f in os.listdir(os.path.join(self.meta_root, 'standard_{}_fold'.format(split_name))):
            data_ix[f.split('.')[0]] = split_name
        return data_ix

    def get_data_ix(self):
        data_ix = {}
        data_ix = self.add_data_ix(data_ix , 'train')
        data_ix = self.add_data_ix(data_ix , 'valid')
        data_ix = self.add_data_ix(data_ix , 'test')
        return data_ix

    def add_sentence(self, label_df):
        text_paths = [os.path.join(self.comb_root,f) for f in os.listdir(self.comb_root)]
        sentences = {}
        for tp in tqdm(text_paths):
            with open(tp,'r') as f:
                text = f.readlines()

            text = [line.strip().split('___') for line in text]
            for txt in text:
                '''
                txt[0] : wav_id
                txt[1] : sent_idx
                txt[2] : start_time
                txt[3] : end_time
                '''
                sel_row = label_df[(label_df['wav_id']==txt[0]) & (label_df['start']==float(txt[2])) & (label_df['end']==float(txt[3]))]
                if len(sel_row) >0: 
                    sel_ix = label_df[(label_df['wav_id']==txt[0]) 
                             & (label_df['start']==float(txt[2])) 
                             & (label_df['end']==float(txt[3]))].index.tolist()[0]
                    sentences[sel_ix] = txt[-1]
        label_df['text'] = label_df.index.map(sentences)
        return label_df


    def add_audio_wav(self, label_df, data_ix, verbose=False): 

        audio_dict = {}
        for f in tqdm(os.listdir(self.audio_root)):

            if f.split('.')[-1] != 'wav':
                '''
                When file is not wav
                '''
                continue

            audio_filename = os.path.join(self.audio_root, f)

            # dialogue
            audio = AudioSegment.from_wav(audio_filename)

            target_ix = os.path.basename(audio_filename).split('.')[0]

            if target_ix not in data_ix :
                '''
                When file is not included in train, valid, test
                '''
                continue

            if label_df[label_df['wav_id'] ==target_ix].shape[0] ==0:
                '''
                When no label
                '''
                continue

            if all(label_df[label_df['wav_id'] ==target_ix]['start'].values >= 0):
                '''
                When start time is negative
                '''

                for start, end in label_df[label_df['wav_id'] ==target_ix][['start', 'end']].values:
                    seg_audio = audio[start*1000:end*1000]
                    samples = seg_audio.get_array_of_samples()
                    arr = np.array(samples).astype(np.float32, order='C') / 32768.0
                    arr_trimed, _ = librosa.effects.trim(arr)

                    #if librosa.get_duration(arr_trimed) > 20:
                    #    logger.info(f'Check {target_ix}-{start}-{end} , that takes long {librosa.get_duration(arr_trimed)}')

                    if verbose:
                        duration = librosa.get_duration(arr)-librosa.get_duration(arr_trimed)
                        if duration > 0 :
                            logger.info(f'Trimmed time as much as {librosa.get_duration(arr)-librosa.get_duration(arr_trimed)} in {target_ix}, {start} ~ {end} ')

                    sel_ix = label_df[(label_df['wav_id']==target_ix) 
                                         & (label_df['start']==start) 
                                         & (label_df['end']==end)].index.tolist()[0]
                    audio_dict[sel_ix] =arr_trimed

        label_df['audio'] = label_df.index.map(audio_dict)

        return label_df
    
    


class Iemocap:
    def __init__(self, label_root, audio_root, text_root):
        self.label_root = label_root
        self.audio_root = audio_root
        self.text_root = text_root
    

    def add_text_dataset(self):
        text_dataset = {}
        for session in self.text_root:
            for txtfile in os.listdir(session):
                with open(os.path.join(session, txtfile), 'r') as f:
                    text = f.readlines()
                for line in text:
                    idx = line.split(':')[0].split()[0]
                    txt = line.split(':')[-1].strip()
                    text_dataset[idx] = txt
        return text_dataset

    def add_audio_dataset(self):
        audio_dataset = {}
        for session in tqdm(self.audio_root):
            wav_folders = [os.path.join(session, subdir) for subdir in os.listdir(session)]
            for situation in wav_folders:
                wav_files = [os.path.join(situation, f) for f in os.listdir(situation) if 'wav' in f ]
                for wav_file in wav_files:
                    idx = os.path.basename(wav_file).split('.')[0]
                    arr, sr = librosa.core.load(wav_file, sr=16000)
                    arr_trimed, _ = librosa.effects.trim(arr)
                    audio_dataset[idx] = arr_trimed
        return audio_dataset

    def add_label_dataset(self):
        label_dict = defaultdict(list)
        for cat in self.label_root:
            label_txts = [os.path.join(cat,f) for f in os.listdir(cat) if 'txt' in f]
            for txtfile in label_txts:
                with open(txtfile, 'r') as f:
                    text = f.readlines()
                for line in text:
                    idx = line.split(':')[0].strip()
                    label = line.split(':')[-1].split(';')[0]
                    label_dict[idx].append(label)
        return label_dict
    
    @staticmethod
    def remove_garbage(x):
        return re.sub('( \[|\[)garbage\]|,', '', x).strip()
    
    @staticmethod
    def name2df(label_dict):
        df = pd.DataFrame(label_dict).T

        name_dict = OrderedDict({l: i for i, l in enumerate(np.unique(df.values))})

        df2lists=[]
        for row in df.values:
            locs = [name_dict[name] for name in row]
            count = {name_dict[i] : 0.0 for i in name_dict}
            for l in locs:
                count[l]+=1 
            df2lists.append(count)

        l_df = pd.DataFrame(df2lists)
        l_df.columns = name_dict.keys()
        l_df.index = label_dict.keys()
        
        return l_df

class Meld:
    def __init__(self, label_root, audio_folders, wav_folders):
        self.label_root = label_root
        self.audio_folders = audio_folders
        self.wav_folders = wav_folders



    def make_wavfiles(self):
        for audio_folder in self.audio_folders:
            audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder)]
            for audio_file in tqdm(audio_files):
                path_part = audio_file.split('/')
                path_part[-2] = 'wav'
                path_part[-1] = path_part[-1].split('.')[0]
                file_path ='/'.join(path_part)

                os.makedirs(os.path.split(file_path)[0], exist_ok=True)

                command = "ffmpeg -i {} -ab 160k -ac 1 -ar 16000 -vn {}.wav".format(audio_file,
                                                                                file_path)
                command = re.sub('새 볼륨', '새\ 볼륨', command)
                subprocess.call(command, shell=True)

            logger.info(f'Completed {audio_folder}.')

    def get_index(self, text):
        dia_idx = re.search('dia(.+?)_', text)
        if dia_idx:
            dia_idx = dia_idx.group(1)
        else:
            raise Exception('There is not diag index!')
        utt_idx = re.search('_utt(.+?).wav', text)
        if utt_idx:
            utt_idx = utt_idx.group(1)
        else:
            raise Exception('There is not utt index!')
        return {'dia' : dia_idx, 'utt': utt_idx}


    def add_audio_dataset(self, df):
        audio_dataset = {}
        for wav_folder in tqdm(self.wav_folders):
            wav_files = [ os.path.join(wav_folder, wav) for wav in os.listdir(wav_folder)]
            for wav_file in wav_files:
                arr, sr = librosa.core.load(wav_file, sr=16000)
                arr_trimed, _ = librosa.effects.trim(arr)

                indexDict = self.get_index(os.path.basename(wav_file))
                index = df[(df['Dialogue_ID'] == int(indexDict['dia'])) &
                           (df['Utterance_ID'] == int(indexDict['utt'])) &
                           (df['part'] == wav_folder.split('/')[-2])].index
                if index.shape[0] > 0:
                    audio_dataset[index[0]] = arr_trimed

        return audio_dataset
    
    @staticmethod
    def concat_parts(data_parts):
        df_lists = []
        for data_part in data_parts:
            df = pd.read_csv(data_part)
            df['part'] = os.path.basename(data_part).split('_')[0]
            df_lists.append(df)
        return pd.concat(df_lists)
    
    @staticmethod
    def drop_na(df):
        return df.dropna()