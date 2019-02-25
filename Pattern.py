import tensorflow as tf;
import numpy as np;
import pandas as pd;
import _pickle as pickle;
from random import shuffle;
from threading import Thread;
from collections import deque;
import time;
import Hyper_Parameters;

class Feeder:
    def __init__(self, batch_Size, max_Queue= 100):
        self.batch_Size = batch_Size;
        self.max_Queue = max_Queue;

        self.Load_Pattern();
        self.Placeholder_Generate();        

        training_Pattern_Generate_Thread = Thread(target=self.Training_Pattern_Generate);
        training_Pattern_Generate_Thread.daemon = True;
        training_Pattern_Generate_Thread.start();

        test_Pattern_Generate_Thread = Thread(target=self.Test_Pattern_Generate);
        test_Pattern_Generate_Thread.daemon = True;
        test_Pattern_Generate_Thread.start();

    def Placeholder_Generate(self):
        '''
        There are two pronunciation placeholders because of test. In test, input is zero array.
        '''
        self.placeholder_Dict = {
            'Is_Training': tf.placeholder(tf.bool, name='Is_Training_Placeholder'),
            'Word': tf.placeholder(tf.int32, shape=(None, self.max_Word_Length), name='Word_Placeholder'),
            'Pronunciation': tf.placeholder(tf.int32, shape=(None, self.max_Pronuniciation_Length), name='Pronunciation_Placeholder')
            }

    def Load_Pattern(self, file_Path= 'Pattern.pickle'):
        with open(file_Path, 'rb') as f:
            load_Dict = pickle.load(f);

        self.training_Pattern_DataFrame = load_Dict['Training'];
        self.test_Pattern_DataFrame = load_Dict['Test'];
        self.letter_Index_Dict = load_Dict['Letter_Index_Dict'];
        self.phoneme_Index_Dict = load_Dict['Phoneme_Index_Dict'];
        self.index_Letter_Dict = {value: key for key, value in self.letter_Index_Dict.items()};
        self.index_Phoneme_Dict = {value: key for key, value in self.phoneme_Index_Dict.items()};

        self.max_Word_Length = len(self.test_Pattern_DataFrame['Word_Pattern.Index'].values[0]);
        self.max_Pronuniciation_Length = len(self.test_Pattern_DataFrame['Pronunciation_Pattern.Index'].values[0]);

    def Training_Pattern_Generate(self):
        self.training_Pattern_Queue = deque();
        
        index_List = [x for x in range(len(self.training_Pattern_DataFrame))]

        while True:
            shuffle(index_List);         
            training_Batch_List = [index_List[start_Index:start_Index+self.batch_Size] for start_Index in range(0, len(index_List), self.batch_Size)];

            current_Index = 0;
            while current_Index < len(training_Batch_List):
                if len(self.training_Pattern_Queue) >= self.max_Queue:
                    time.sleep(0.1);
                    continue;

                batch_Pattern_Dataframe = self.training_Pattern_DataFrame.loc[training_Batch_List[current_Index]]    #batch patterns dataframe

                new_Feed_Dict = {
                    self.placeholder_Dict['Is_Training']: True,
                    self.placeholder_Dict['Word']: np.stack(batch_Pattern_Dataframe['Word_Pattern.Index'].values),
                    self.placeholder_Dict['Pronunciation']: np.stack(batch_Pattern_Dataframe['Pronunciation_Pattern.Index'].values)
                    }

                self.training_Pattern_Queue.append(new_Feed_Dict);

                current_Index += 1;

    def Get_Training_Pattern(self):
        while True:
            if len(self.training_Pattern_Queue) > 0:
                break;
            print('No Data')
            time.sleep(0.1);

        return self.training_Pattern_Queue.popleft();

    def Test_Pattern_Generate(self):
        index_List = list(range(len(self.test_Pattern_DataFrame['Pronunciation_Pattern.Index'])));
        test_Batch_List = [index_List[start_Index:start_Index+self.batch_Size] for start_Index in range(0, len(index_List), self.batch_Size)];

        self.test_Pattern_List = [];
        for test_Batch in test_Batch_List:
            batch_Pattern_Dataframe = self.test_Pattern_DataFrame.loc[test_Batch]    #batch patterns dataframe
            self.test_Pattern_List.append({
                self.placeholder_Dict['Is_Training']: False,
                self.placeholder_Dict['Word']: np.stack(batch_Pattern_Dataframe['Word_Pattern.Index'].values),
                self.placeholder_Dict['Pronunciation']: np.stack(batch_Pattern_Dataframe['Pronunciation_Pattern.Index'].values)
                })

    def Get_Test_Pattern(self):        
        while not hasattr(self, 'test_Pattern_List'):            
            time.sleep(0.1);
        return self.test_Pattern_List;

    def Get_Inference_Pattern(self, word_List):
        sparse_Word_Pattern_List = [];
        for word in word_List:
            word = ['<S>'] + list(word) + ['<E>'] + ['<P>'] * (self.max_Word_Length - 2 - len(word));   #-2 is about <S> and <E>
            new_Sparse_Pattern = np.array([self.letter_Index_Dict[letter] for letter in word]).astype(np.int32);
            sparse_Word_Pattern_List.append(new_Sparse_Pattern);

        inference_Pattern = {
            self.placeholder_Dict['Is_Training']: False,
            self.placeholder_Dict['Word']: np.stack(sparse_Word_Pattern_List)
            }        
        return inference_Pattern

    def Index_to_Phoneme(self, index_Array):        
        return ''.join([
            self.index_Phoneme_Dict[index]
            for index in index_Array
            if not self.index_Phoneme_Dict[index] in ['<P>', '<S>', '<E>']
            ])