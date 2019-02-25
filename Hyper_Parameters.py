import tensorflow as tf

Save_Path = 'E:\Self_Attention_Tutorial_Result';
Batch_Size = 1024;
Test_Timing = 500;

#Embedding_Size == Attention_Size * Head_Size
Embedding_Size = 256; #512;
Num_Blocks = 6; #6;
Attention_Size =  64;
Head_Size = 4;  #8
Feedforward_Size_List = [512]; #[2048];
Dropout_Rate = 0.5;