#Refer: https://github.com/Kyubyong/transformer

import tensorflow as tf;
import numpy as np;
import matplotlib.pyplot as plt;
import os, time;
from Pattern import Feeder;
from Module import *;
import Hyper_Parameters;


class Self_Attention_Model:
    def __init__(self):
        self.tf_Session = tf.Session();
        self.feeder = Feeder(Hyper_Parameters.Batch_Size);
        self.Tensor_Generate();
        self.tf_Saver = tf.train.Saver(max_to_keep=5);

        os.makedirs(Hyper_Parameters.Save_Path, exist_ok= True);

        self.Restore();

    def Tensor_Generate(self):
        placeholder_Dict = self.feeder.placeholder_Dict;

        with tf.variable_scope('encoder') as scope:
            #Index and position embedding
            encoder_Embedding_Tensor = Embedding(
                inputs= placeholder_Dict['Word'],
                id_size= len(self.feeder.letter_Index_Dict),
                embedding_size = Hyper_Parameters.Embedding_Size,
                trainable= True
                )

            #Dropout
            encoder_Embedding_Tensor = tf.layers.dropout(
                encoder_Embedding_Tensor,
                rate= Hyper_Parameters.Dropout_Rate,
                training= placeholder_Dict['Is_Training'],
                name='dropout'
                )

            #Block
            encoder_Tensor = Encoder_Block(
                encoder_tensor= encoder_Embedding_Tensor,
                num_blocks= Hyper_Parameters.Num_Blocks,
                attention_size= Hyper_Parameters.Attention_Size,
                head_size= Hyper_Parameters.Head_Size,
                feedforward_size_list= Hyper_Parameters.Feedforward_Size_List,
                dropout_Rate= Hyper_Parameters.Dropout_Rate,
                is_Training= placeholder_Dict['Is_Training']
                )

        with tf.variable_scope('decoder') as scope:
            #Index and position embedding
            decoder_Embedding_Tensor = Embedding(
                inputs= placeholder_Dict['Pronunciation'],
                id_size= len(self.feeder.phoneme_Index_Dict),
                embedding_size = Hyper_Parameters.Embedding_Size,
                trainable= True
                )

            #Dropout
            decoder_Embedding_Tensor = tf.layers.dropout(
                decoder_Embedding_Tensor,
                rate= Hyper_Parameters.Dropout_Rate,
                training= placeholder_Dict['Is_Training'],
                name='dropout'
                )

        #For train
        with tf.variable_scope('decoder') as scope:
            ##Block
            decoder_Tensor, _ = Decoder_Block(
                decoder_tensor= decoder_Embedding_Tensor,
                encoder_tensor= encoder_Tensor,
                num_blocks= Hyper_Parameters.Num_Blocks,
                attention_size= Hyper_Parameters.Attention_Size,
                head_size= Hyper_Parameters.Head_Size,
                feedforward_size_list= Hyper_Parameters.Feedforward_Size_List,
                dropout_Rate= Hyper_Parameters.Dropout_Rate,
                is_Training= placeholder_Dict['Is_Training']
                )
            
            #Linear projection
            logits = tf.layers.dense(
                decoder_Tensor,
                len(self.feeder.phoneme_Index_Dict),
                name='logits'
                )

        with tf.variable_scope('loss') as scope:
            batch_Size = tf.shape(placeholder_Dict['Pronunciation'])[0];
            shifted_Target = tf.concat([placeholder_Dict['Pronunciation'][:, 1:], tf.zeros((batch_Size, 1), dtype=tf.int32)], axis=-1);
        
            #Padding mask can be removed. I think this is to allow the model to focus on real information, not padding.
            padding_Mask = tf.cast(tf.not_equal(shifted_Target, self.feeder.phoneme_Index_Dict['<P>']), tf.float32);

            #Accuracy
            predictions = tf.argmax(logits, axis= -1, output_type=tf.int32);
            accuracy_Map = tf.cast(tf.equal(predictions, shifted_Target), tf.float32);
            comparing = tf.equal(
                tf.reduce_sum(accuracy_Map * padding_Mask, axis = -1),
                tf.reduce_sum(padding_Mask, axis=-1)
                )
            train_Accuracy = tf.reduce_mean(tf.cast(comparing, tf.float32))

            smoothed_Labels = Label_Smoothing(tf.one_hot(
                shifted_Target,
                depth= len(self.feeder.phoneme_Index_Dict)
                ))
            
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels= smoothed_Labels,
                logits= logits,
                )
            loss = tf.reduce_sum(loss * padding_Mask) / tf.reduce_sum(padding_Mask);

            global_Step = tf.Variable(0, name='global_step', trainable = False);

            #Noam decay
            step = tf.cast(global_Step + 1, dtype=tf.float32);
            warmup_Steps = 4000.0;
            learning_Rate = (Hyper_Parameters.Attention_Size * Hyper_Parameters.Head_Size) ** -0.5 * \
                tf.minimum(step * warmup_Steps**-1.5, step**-0.5)

            #Adam
            optimizer = tf.train.AdamOptimizer(learning_Rate, beta1= 0.9, beta2= 0.98, epsilon= 1e-09);
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            clipped_Gradients, global_Norm = tf.clip_by_global_norm(gradients, 1.0)
            optimize = optimizer.apply_gradients(zip(clipped_Gradients, variables), global_step=global_Step)
            
        #For test
        with tf.variable_scope('decoder', reuse= True) as scope:
            batch_Size = tf.shape(encoder_Tensor)[0];

            initial_Input_Tensor = tf.ones(shape=(batch_Size, 1), dtype=tf.int32) * self.feeder.phoneme_Index_Dict['<S>'];
            decoder_Input_Tensor = initial_Input_Tensor;
            ##Block
            for _ in range(self.feeder.max_Pronuniciation_Length):
                decoder_Embedding_Tensor = Embedding(
                    inputs= decoder_Input_Tensor,
                    id_size= len(self.feeder.phoneme_Index_Dict),
                    embedding_size = Hyper_Parameters.Embedding_Size,
                    trainable= True
                    )

                decoder_Tensor, attention_History_Tensor = Decoder_Block(
                    decoder_tensor= decoder_Embedding_Tensor,
                    encoder_tensor= encoder_Tensor,
                    num_blocks= Hyper_Parameters.Num_Blocks,
                    attention_size= Hyper_Parameters.Attention_Size,
                    head_size= Hyper_Parameters.Head_Size,
                    feedforward_size_list= Hyper_Parameters.Feedforward_Size_List,
                    dropout_Rate= Hyper_Parameters.Dropout_Rate,
                    is_Training= placeholder_Dict['Is_Training']
                    )

                #Linear projection
                logits = tf.layers.dense(
                    decoder_Tensor,
                    len(self.feeder.phoneme_Index_Dict),
                    name='logits'
                    )

                predictions = tf.argmax(logits, axis= -1, output_type=tf.int32);                
                decoder_Input_Tensor = tf.concat([initial_Input_Tensor, predictions], axis=-1);

        with tf.variable_scope('test') as scope:            
            predictions = tf.argmax(logits, axis= -1, output_type=tf.int32);            
            accuracy_Map = tf.cast(tf.equal(predictions, shifted_Target), tf.float32);
            comparing = tf.equal(
                tf.reduce_sum(accuracy_Map * padding_Mask, axis = -1),
                tf.reduce_sum(padding_Mask, axis=-1)
                )
            test_Accuracy = tf.cast(comparing, tf.float32)

        self.training_Tensor_Dict = {
            'Glaobal_Step': global_Step,
            'Learning_Rate': learning_Rate,
            'Loss': loss,
            'Accuracy': train_Accuracy,
            'optimize': optimize
            }

        self.test_Tensor_Dict = {
            'Glaobal_Step': global_Step,
            'Accuracy': test_Accuracy,
            'Predictions': predictions
            }

        self.inference_Tensor_Dict = {
            'Global_Step': global_Step,
            'Predictions': predictions,
            'Attention_Histories': attention_History_Tensor
            }

        self.tf_Session.run(tf.global_variables_initializer());

    def Restore(self):        
        checkpoint_Path = os.path.join(Hyper_Parameters.Save_Path, 'Checkpoint').replace('\\', '/');
        os.makedirs(checkpoint_Path, exist_ok= True);

        checkpoint_Path = tf.train.latest_checkpoint(checkpoint_Path);
        print('Lastest checkpoint:', checkpoint_Path);

        if checkpoint_Path is None:
            print('There is no checkpoint');
        else:
            self.tf_Saver.restore(self.tf_Session, checkpoint_Path);
            print('Checkpoint \'', checkpoint_Path, '\' is loaded');

    def Train(self, test_Timing= Hyper_Parameters.Test_Timing):
        self.Test();
        while True:
            start_Time = time.time();
            result_Dict = self.tf_Session.run(
                fetches= self.training_Tensor_Dict,
                feed_dict= self.feeder.Get_Training_Pattern()
                )
            print('Spent_Time: {:.5f}\t\tGlobal step: {}\t\tLearning rate: {:.5f}\t\tLoss: {:.5f}\t\tAccuracy: {:.5f}'.format(
                time.time() - start_Time,
                result_Dict['Glaobal_Step'],
                result_Dict['Learning_Rate'],
                result_Dict['Loss'],
                result_Dict['Accuracy']
                ))

            if test_Timing > 0 and result_Dict['Glaobal_Step'] % test_Timing == 0:
                self.Checkpoint_Save(result_Dict['Glaobal_Step']);
                self.Test();
                

    def Checkpoint_Save(self, global_Step):
        checkpoint_Path = os.path.join(Hyper_Parameters.Save_Path, 'Checkpoint').replace('\\', '/');
        os.makedirs(checkpoint_Path, exist_ok= True);
        self.tf_Saver.save(self.tf_Session, os.path.join(checkpoint_Path, 'Checkpoint').replace('\\', '/'), global_step=global_Step);
        print('Checkpoint saved');

    def Test(self):
        test_Pattern_List = self.feeder.Get_Test_Pattern();
        result_Dict_List = [];

        for test_Pattern in test_Pattern_List:
            result_Dict_List.append(self.tf_Session.run(
                fetches= self.test_Tensor_Dict,
                feed_dict= test_Pattern
                ))

        result_Dict = {
            'Glaobal_Step': result_Dict_List[-1]['Glaobal_Step'],
            'Accuracy': np.hstack([x['Accuracy'] for x in result_Dict_List]),
            'Predictions': np.vstack([x['Predictions'] for x in result_Dict_List]),
            }

        print('Glaobal Step {}\t\tTest Accuracy: {:05f}'.format(
            result_Dict['Glaobal_Step'],
            np.mean(result_Dict['Accuracy'])
            ))

        prediction_List = [self.feeder.Index_to_Phoneme(prediction) for prediction in result_Dict['Predictions']];
        export_Zip = zip(
            ['Word'] + self.feeder.test_Pattern_DataFrame['Word'].tolist(),
            ['Pronunciation'] + self.feeder.test_Pattern_DataFrame['Pronunciation'].tolist(),
            ['Prediction'] + prediction_List,
            ['Accuracy'] + result_Dict['Accuracy'].tolist()
            )
        
        test_Path = os.path.join(Hyper_Parameters.Save_Path, 'Test').replace('\\', '/');
        os.makedirs(test_Path, exist_ok= True);
        with open(os.path.join(test_Path, '{}.txt'.format(str(result_Dict['Glaobal_Step']))).replace('\\', '/'), 'w') as f:
            f.write('\n'.join(['\t'.join([word, pronunciation, prediction, str(accuracy)]) for word, pronunciation, prediction, accuracy in export_Zip]));

        if result_Dict['Glaobal_Step'] > 9999:
            assert False;

    def Inference(self, word_List, file='Inference.txt'):
        result_Dict = self.tf_Session.run(
            fetches= self.inference_Tensor_Dict,
            feed_dict= self.feeder.Get_Inference_Pattern(word_List)
            )

        export_Zip = zip(
            word_List,
            [self.feeder.Index_to_Phoneme(prediction) for prediction in result_Dict['Predictions']]
            )

        inference_Path = os.path.join(Hyper_Parameters.Save_Path, 'Inference').replace('\\', '/');
        os.makedirs(inference_Path, exist_ok= True);

        with open(os.path.join(inference_Path, file).replace('\\', '/'), 'w') as f:
            f.write('\n'.join(['\t'.join([word, prediction]) for word, prediction in export_Zip]));

        for word, prediction, attention_History in zip(word_List, result_Dict['Predictions'], result_Dict['Attention_Histories']):            
            pronunciation = self.feeder.Index_to_Phoneme(prediction);
            
            fig = plt.figure(figsize=(10 * Hyper_Parameters.Num_Blocks, 10 * Hyper_Parameters.Head_Size));
            for block_Index in range(Hyper_Parameters.Num_Blocks):
                for head_Index in range(Hyper_Parameters.Head_Size):
                    plt.subplot(
                        Hyper_Parameters.Num_Blocks,
                        Hyper_Parameters.Head_Size,
                        Hyper_Parameters.Head_Size * block_Index + head_Index + 1   #start from 1
                        )

                    plt.imshow(
                        np.transpose(attention_History[block_Index, head_Index][:len(word), :len(pronunciation)]),
                        vmin=0.0,
                        vmax=1.0,
                        cmap="viridis",
                        aspect='auto',
                        origin='lower',
                        interpolation='none'
                        )
                    plt.xlabel("Word");
                    plt.ylabel("Pronunciation");

                    plt.gca().set_xticks(range(len(word)));
                    plt.gca().set_xticklabels([x for x in word]);
                    plt.gca().set_yticks(range(len(pronunciation)));
                    plt.gca().set_yticklabels([x for x in pronunciation]); 
                    plt.colorbar();
                    plt.title("Step: {}    Block: {}    Head: {}".format(result_Dict['Global_Step'], block_Index, head_Index));

            plt.savefig(
                os.path.join(inference_Path, 'S_{}.W_{}.png'.format(result_Dict['Global_Step'], word)).replace('\\', '/'),
                bbox_inches='tight'
                )
            plt.close(fig)

if __name__ == '__main__':
    self_Attention_Model = Self_Attention_Model();    
    #self_Attention_Model.Train();
    self_Attention_Model.Inference(word_List = ['school'])
   