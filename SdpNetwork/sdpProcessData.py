#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 11:43:04 2018

@author: llq
"""
import numpy as np
import re

class sdpProcessData:
    
    def __init__(self):
        """
        1.Two sdp file in train data
        """
        self.e1_sdp_train_file="./sdpData/train/train_e1_SDP.txt"
        self.e2_sdp_train_file="./sdpData/train/train_e2_SDP.txt"
        self.train_e1_sdp_pos_file="./sdpData/train/train_e1_sdp_pos.txt"
        self.train_e2_sdp_pos_file="./sdpData/train/train_e2_sdp_pos.txt"
        
        """
        2.Two sdp file in test data
        """
        self.e1_sdp_test_file="./sdpData/test/test_e1_SDP.txt"
        self.e2_sdp_test_file="./sdpData/test/test_e2_SDP.txt"
        self.test_e1_sdp_pos_file="./sdpData/test/test_e1_sdp_pos.txt"
        self.test_e2_sdp_pos_file="./sdpData/test/test_e2_sdp_pos.txt"
 
        """
        2.Word2vec
        """
        #word2vec file
        self.output_vector_filename=r"../data/GoogleNews_vec.txt"
        #Dictory:store the word and vector
        self.dict_word_vec={}
        #Vector Size
        self.vector_size=300
        
        """
        3.(1)Initial max sentence length.
          (2)Store the label id.
        """
        self.label2id_txt="../processData/label2id.txt"
        self.max_length_sen=31
        #label id value: Change the label to id.And 10 classes number(0-9)
        self.label2id={}
                
        """
        4.(1)Position:initial the position vector
        (2)Dependencies:initial the dependencies vector
        """
        self.pos2vec_length=20
        self.pos2vec_init=np.random.normal(size=(41,self.pos2vec_length),loc=0,scale=0.05)
        
        self.dep_length=20
        self.dep_file="./sdpData/dependencies.txt"
        self.dep_dict_init={}

        """
        5.Process training data 
        """
        #training sentence
        self.training_sen_number=8000
        #training data,this format:8000*105*300
        self.training_word_vec3D=np.empty((self.training_sen_number,self.max_length_sen,self.vector_size))
        self.training_word_vec3D_reverse=np.empty((self.training_sen_number,self.max_length_sen,self.vector_size))
        #Training data:word and position vector.this format:8000*105*320
        self.training_word_pos_vec3D=np.empty((self.training_sen_number,self.max_length_sen,self.vector_size+self.pos2vec_length*2))
        #training label
        self.training_label=np.empty((self.training_sen_number)).astype(int)
        #training label
        self.training_label_reverse=np.empty((self.training_sen_number)).astype(int)
        #record the length of sentence 
        self.training_sen_length=np.empty((self.training_sen_number,1))
        #train denpendecies
        self.train_word_dep_file="./sdpData/train/train_dependencies.txt"
        #SDP linked to denpendecies
        self.train_sdp_link_dep_file="./sdpData/train/train_sdp_link_dep.txt"
        self.train_sdp_link_dep_reverse_file="./sdpData/train_reverse/train_sdp_link_dep_reverse.txt"
        #label in train data
        self.train_label_store_filename=r"../processData/train_label.txt"
        #save reverse label in txt
        self.train_label_reverse_txt="../processData/train_label_reverse.txt"
        
        """
        6.Process testing data
        """
        #Testing sentence
        self.testing_sen_number=2717
        #Testing data,this format:2717*105*300
        self.testing_word_vec3D=np.empty((self.testing_sen_number,self.max_length_sen,self.vector_size))
        #Testing data:word and position vector.this format:8000*105*320
        self.testing_word_pos_vec3D=np.empty((self.testing_sen_number,self.max_length_sen,self.vector_size+self.pos2vec_length*2))
        #Testing label
        self.testing_label=np.empty((self.testing_sen_number)).astype(int)
        #record the length of sentence 
        self.testing_sen_length=np.empty((self.testing_sen_number,1))
        #train denpendecies
        self.test_word_dep_file="./sdpData/test/test_dependencies.txt"
        #SDP linked to denpendecies
        self.test_sdp_link_dep_file="./sdpData/test/test_sdp_link_dep.txt"
        #label in train data
        self.test_label_store_filename=r"../processData/test_label.txt"
        
    def dict_word2vec(self):
        """
        When create Process_data,must exec this function.
        Initial dict_word_vec.
        """
        #put the vector in the dictionary
        with open(self.output_vector_filename,"r") as f:
            i=0
            for lines in f.readlines():
                if(i==0):
                    i=i+1
                    continue
                lines_split=lines.split(" ")
                keyword=lines_split[0]
                lines_split=map(float,lines_split[1:-1])
                self.dict_word_vec[keyword]=lines_split
        
        #Set value in "BLANK",its size is 300
        self.dict_word_vec["BLANK"]=np.random.normal(size=self.vector_size,loc=0,scale=0.05)
               
        
    def label2id_init(self):
        """
        When create Process_data,must exec this function.
        Change the traing label value to id. 
        """
        with open(self.label2id_txt,"r") as f:
            for lines in f.readlines():
                lines=lines.strip("\r\n").split()
                self.label2id[lines[0]]=lines[1]
    
    def dep_vec_init(self):
        """
        Init the dependencies vector
        """
        with open(self.dep_file,"r") as f:
            for lines in f.readlines():
                lines=lines.strip("\r\n").split()
                self.dep_dict_init[lines[0]]=np.random.normal(size=self.dep_length,loc=0,scale=0.05)
        
        #insert "no_dep"
        self.dep_dict_init["no_dep"]=np.random.normal(size=self.dep_length,loc=0,scale=0.05)
        
    def combine_sdp(self,fore_sdp_file,back_sdp_file):
        """
        Combine two SDP:
            return con_spd:[[sentence word],[sentence word]....]
        """
        #Stroe the first sdp
        first_sdp=[]
        #Stroe the second sdp
        second_sdp=[]
        #Stroe the combine sdp
        con_spd=[]

        with open(fore_sdp_file,"r") as file:
            for line in file.readlines():
                first_sdp.append(list(reversed(line.split(" ")[0:-1])))
                    
        with open(back_sdp_file,"r") as file:
            for line in file.readlines():
                second_sdp.append(line.split(" ")[1:-1])

        #combine "first_sdp" and "second_sdp"
        for i in range(len(first_sdp)):
            con_spd.append(first_sdp[i]+second_sdp[i])
            
        return con_spd
    
        
    def word_dep(self,dep_file,fore_sdp_pos_file,back_sdp_pos_file,sdp_dep_file):
        """
        Save the dependencies linked to SDP
        """
        #Store the fore position
        fore_pos=[]
        #Store the back position
        back_pos=[]
        #Store dependecies which is linked to SDP
        sdp_dep=open(sdp_dep_file,"w")
        
        with open(fore_sdp_pos_file,"r") as file:
            for line in file.readlines():
                fore_pos.append(line.split(" ")[:-1])
        
        with open(back_sdp_pos_file,"r") as file:
            for line in file.readlines():
                back_pos.append(line.split(" ")[:-1])
                
        #replace this punctuation
        r='-[0-9]+'
        with open(dep_file,"r") as f:
            sentence_id=0
            for lines in f.readlines():
                #Dict:Save "word position" and "dependencies"
                word_pos_dep_dict={}
                #save dep of fore_pos,because we must reverse it
                fore_dep_reverse=[]
                word_dep=lines.split("\t")[:-1]
                
                #look for the dependencies in (fore_word,back_word)
                for word in word_dep:
                    dependecy=word.split(", ")[0].split("(")[0]
                    fore_word=word.split(", ")[0].split("(")[1]
                    back_word=word.split(", ")[1].split(")")[0]
                    
                    #get the number from "xxx-2"
                    fore_word=re.search(r,fore_word).group().replace("-","")
                    back_word=re.search(r,back_word).group().replace("-","")
                    
                    word_pos_dep_dict[fore_word+"->"+back_word]=dependecy
                
                #get the dependencies
                for i in range(len(fore_pos[sentence_id])-1):
                    try:
                        fore_dep_reverse.append(word_pos_dep_dict[fore_pos[sentence_id][i]+"->"+fore_pos[sentence_id][i+1]])
                    except:
                        fore_dep_reverse.append("root")
#                        print sentence_id
                        
                #reverse the dep in fore_word
                fore_dep_reverse=list(reversed(fore_dep_reverse))
                for dep in fore_dep_reverse:
                    sdp_dep.write(dep+"\t")
                    
                #write "ROOT"
                sdp_dep.write("root"+"\t")
                
                for i in range(len(back_pos[sentence_id])-1):
                    try:
                        sdp_dep.write(word_pos_dep_dict[back_pos[sentence_id][i]+"->"+back_pos[sentence_id][i+1]]+"\t")
                    except:
                        sdp_dep.write("root"+"\t")
#                        print sentence_id
                sdp_dep.write("\n")

                sentence_id+=1
        sdp_dep.close()
        
        
    def word_interval_dep(self,dep_file,fore_sdp_pos_file,back_sdp_pos_file,sdp_dep_file,fore_sdp_file,back_sdp_file):
        """
        Save the dependencies linked to SDP:
            word1,dep1,word2,dep2.....
        """
        #Store the fore position
        fore_pos=[]
        #Store the back position
        back_pos=[]
        #Store the fore word
        fore_word=[]
        #Store the back word
        back_word=[]
        #Store dependecies which is linked to SDP
        sdp_dep=open(sdp_dep_file,"w")
        
        with open(fore_sdp_pos_file,"r") as file:
            for line in file.readlines():
                fore_pos.append(line.split(" ")[:-1])
        
        with open(back_sdp_pos_file,"r") as file:
            for line in file.readlines():
                back_pos.append(line.split(" ")[:-1])
                
        #replace this punctuation
        r='-[0-9]+'
        with open(dep_file,"r") as f:
            sentence_id=0
            for lines in f.readlines():
                #Dict:Save "word position" and "dependencies"
                word_pos_dep_dict={}
                #save dep of fore_pos,because we must reverse it
                fore_dep_reverse=[]
                word_dep=lines.split("\t")[:-1]
                
                #look for the dependencies in (fore_word,back_word)
                for word in word_dep:
                    dependecy=word.split(", ")[0].split("(")[0]
                    fore_word=word.split(", ")[0].split("(")[1]
                    back_word=word.split(", ")[1].split(")")[0]
                    
                    #get the number from "xxx-2"
                    fore_word=re.search(r,fore_word).group().replace("-","")
                    back_word=re.search(r,back_word).group().replace("-","")
                    
                    word_pos_dep_dict[fore_word+"->"+back_word]=dependecy
                
                #get the dependencies
                for i in range(len(fore_pos[sentence_id])-1):
                    try:
                        fore_dep_reverse.append(word_pos_dep_dict[fore_pos[sentence_id][i]+"->"+fore_pos[sentence_id][i+1]])
                    except:
                        fore_dep_reverse.append("root")
#                        print sentence_id
                        
                #reverse the dep in fore_word
                fore_dep_reverse=list(reversed(fore_dep_reverse))
                for dep in fore_dep_reverse:
                    sdp_dep.write(dep+"\t")
                    
                #write "ROOT"
                sdp_dep.write("root"+"\t")
                
                for i in range(len(back_pos[sentence_id])-1):
                    try:
                        sdp_dep.write(word_pos_dep_dict[back_pos[sentence_id][i]+"->"+back_pos[sentence_id][i+1]]+"\t")
                    except:
                        sdp_dep.write("root"+"\t")
#                        print sentence_id
                sdp_dep.write("\n")

                sentence_id+=1
        sdp_dep.close()
        
        
    def pos_embed(self,x):
        """
        embedding the position 
        """
        if x < -20:
        	return 0
        if x >= -20 and x <= 20:
        	return x+20
        if x > 20:
        	return 40
    
    '''
    #each sentences is tre number
    
    def embedding_lookup(self,con_spd,word_vec3D,sen_number,sdp_link_dep_file):
        """
        1.con_spd:put sentence in this.format:[[sentence1],[sentence2]]
        2.word_vec3D:get each word vector,and Make data to this format:8000*max_length_sen*300. 
            In 105*300,the first dim is word;the sencond dim is vector
        3.word_pos_vec3D:has "word vector" and "position vector".
            this format is N*max_length_sen*340,(N has two value "8000" and "2717")
        """
        
        #sen_length:length of sentence         
        sen_length=[]
        #store the position id
#        pos_id=np.zeros((sen_number,self.max_length_sen,2))
        pos_id=[]
        
        sentence_id=0
        for sentences in con_spd:
            
            #Position number of e1 and e2
            pos_e1=0
            pos_e2=len(sentences)
            
            #store the original length of sentence
            sen_length.append(len(sentences))
            
            #check the length in "con_spd":
            #   If length of "con_spd"<max_length_sen, set "BLANK".
            #   If length of "con_spd">max_length_sen, truncation it.
#            if(len(sentences)<self.max_length_sen):
#                for i in range(self.max_length_sen-len(sentences)):
#                        con_spd[sentence_id].append('BLANK')
#            if(len(sentences)>self.max_length_sen):
#                for i in range(len(sentences)-self.max_length_sen):
#                        con_spd[sentence_id].pop()
                        
            #Get the "entity word"-"other word" in this.
            #pos_id format:N*sen_number*2,(N has two value "8000" and "2717")
            #And sen_number(word)*2(id):
            #     [pos_id1,pos_id2],
            #     [pos_id1,pos_id2],
            #     [pos_id1,pos_id2],
            #     [pos_id1,pos_id2],
            word_pos_id=[]
            for i in range(len(con_spd[sentence_id])):
#                pos_id[sentence_id][i]=\
#                    np.array([self.pos_embed(i-pos_e1),self.pos_embed(i-pos_e2)])
                word_pos_id.append([self.pos_embed(i-pos_e1),self.pos_embed(i-pos_e2)])
            pos_id.append(word_pos_id)
  
            sentence_id+=1
        
        #1.Set the "position word" to vector.
        #pos_vec:N(sentence)*length_sen(word)*pos2vec_length(position vector),(N has two value "8000" and "2717")
#        pos_vec=np.zeros((sen_number,self.max_length_sen,self.pos2vec_length*2))
        pos_vec=[]    
        sentence_id=0
        for word in pos_id:
            i=0
            word_pos_vec=[]
            for pos_num in word:
#                pos_vec[sentence_id][i]=np.hstack\
#                    ((self.pos2vec_init[int(pos_num[0])],self.pos2vec_init[int(pos_num[1])]))
#                i=i+1
                word_pos_vec.append(np.hstack\
                   ((self.pos2vec_init[int(pos_num[0])],self.pos2vec_init[int(pos_num[1])])))
            pos_vec.append(word_pos_vec)
            sentence_id=sentence_id+1
          
        #2.Find the word vector in dict_word_vec.
        #Make data to this format:N*sen_number*300,(N has two value "8000" and "2717")
        #In sen_number*300,the first dim is "word";the sencond dim is "vector"
        sen_vec3D=[]
        sentence_id=0
        for sentences in con_spd:
            #store word vector in each sentences
            sen_word_vec3D=[]
            for words in sentences:
                #find word in dict_word_vec
                if(self.dict_word_vec.has_key(words)):
                    sen_word_vec3D.append(self.dict_word_vec[words])
                else:
                    self.dict_word_vec[words]=np.random.normal(size=(1,self.vector_size),loc=0,scale=0.05)
                    sen_word_vec3D.append(self.dict_word_vec[words])
            sentence_id=sentence_id+1
            sen_vec3D.append(sen_word_vec3D)
                  
        #3.Set dependencies vector
        #Store the dep in each sentences
        dep_list=[]
        with open(sdp_link_dep_file,"r") as f:
            sentence_id=0
            for lines in f.readlines():
                lines=lines.split("\t")[:-1]
                dep_list.append(lines)
                
#                #check the length in "dep_list": 
#                if(len(lines)<self.max_length_sen):
#                    for i in range(self.max_length_sen-len(lines)):
#                            dep_list[sentence_id].append('no_dep')
#                if(len(lines)>self.max_length_sen):
#                    for i in range(len(lines)-self.max_length_sen):
#                            dep_list[sentence_id].pop()
                            
                sentence_id+=1
        
        #Dependencies in each sentences.
        #It's format (N*length_sen*dep_length):
        #   "N" is length of sentences in train data or test data
        #   "length_sen" is word number in each sentences
        #   "dep_length" is 20
        dep_vec=[]      
        for sen in dep_list:
            dep_word_vec=[]
            for dep in sen:
                dep_word_vec.append(self.dep_dict_init[dep])
            dep_vec.append(dep_word_vec)
              
        #At last,concatenate "sen_vec3D" and "pos_vec" and "dep_vec"
        #      =>self.word_pos_vec3D
        word_pos_vec3D=[]
        for sen_id in range(len(sen_vec3D)):
            word_pos_in_sen=[]
            for word_id in range(len(sen_vec3D[sen_id])):
                word_pos_in_sen.append(np.concatenate(
                   (np.float32(np.reshape(sen_vec3D[sen_id][word_id],(1,-1))),\
                    np.float32(np.reshape(pos_vec[sen_id][word_id],(1,-1))),\
                    np.float32(np.reshape(dep_vec[sen_id][word_id],(1,-1)))),axis=1))
                
            word_pos_vec3D.append(word_pos_in_sen)

        return word_pos_vec3D,sen_length,pos_vec
    '''
    
    def embedding_lookup(self,con_spd,word_vec3D,sen_number,sdp_link_dep_file):
        """
        1.con_spd:put sentence in this.format:[[sentence1],[sentence2]]
        2.word_vec3D:get each word vector,and Make data to this format:8000*max_length_sen*300. 
            In 105*300,the first dim is word;the sencond dim is vector
        3.word_pos_vec3D:has "word vector" and "position vector".
            this format is N*max_length_sen*340,(N has two value "8000" and "2717")
        """
        
        #sen_length:length of sentence         
        sen_length=[]
        #store the position id
        pos_id=np.zeros((sen_number,self.max_length_sen,2))
        
        sentence_id=0
        for sentences in con_spd:
            
            #Position number of e1 and e2
            pos_e1=0
            pos_e2=len(sentences)
            
            #store the original length of sentence
            sen_length.append(len(sentences))
            
            #check the length in "con_spd":
            #   If length of "con_spd"<max_length_sen, set "BLANK".
            #   If length of "con_spd">max_length_sen, truncation it.
            if(len(sentences)<self.max_length_sen):
                for i in range(self.max_length_sen-len(sentences)):
                        con_spd[sentence_id].append('BLANK')
            if(len(sentences)>self.max_length_sen):
                for i in range(len(sentences)-self.max_length_sen):
                        con_spd[sentence_id].pop()
                        
            #Get the "entity word"-"other word" in this.
            #pos_id format:N*sen_number*2,(N has two value "8000" and "2717")
            #And sen_number(word)*2(id):
            #     [pos_id1,pos_id2],
            #     [pos_id1,pos_id2],
            #     [pos_id1,pos_id2],
            #     [pos_id1,pos_id2],
            for i in range(len(con_spd[sentence_id])):
                pos_id[sentence_id][i]=\
                    np.array([self.pos_embed(i-pos_e1),self.pos_embed(i-pos_e2)])
  
            sentence_id+=1
        
        #1.Set the "position word" to vector.
        #pos_vec:N(sentence)*length_sen(word)*pos2vec_length(position vector),(N has two value "8000" and "2717")
        pos_vec=np.zeros((sen_number,self.max_length_sen,self.pos2vec_length*2))   
        sentence_id=0
        for word in pos_id:
            i=0
            for pos_num in word:
                pos_vec[sentence_id][i]=np.hstack\
                    ((self.pos2vec_init[int(pos_num[0])],self.pos2vec_init[int(pos_num[1])]))
                i=i+1

            sentence_id=sentence_id+1
          
        #2.Find the word vector in dict_word_vec.
        #Make data to this format:N*sen_number*300,(N has two value "8000" and "2717")
        #In sen_number*300,the first dim is "word";the sencond dim is "vector"
        sentence_id=0
        for sentences in con_spd:
            word_id=0
            for words in sentences:
                #find word in dict_word_vec
                if(self.dict_word_vec.has_key(words)):
                    word_vec3D[sentence_id][word_id]=self.dict_word_vec[words]
                else:
                    self.dict_word_vec[words]=np.random.normal(size=(1,self.vector_size),loc=0,scale=0.05)
                    word_vec3D[sentence_id][word_id]=self.dict_word_vec[words]
                
                word_id+=1
            sentence_id=sentence_id+1
                  
        #3.Set dependencies vector
        #Store the dep in each sentences
        dep_list=[]
        with open(sdp_link_dep_file,"r") as f:
            sentence_id=0
            for lines in f.readlines():
                lines=lines.split("\t")[:-1]
                dep_list.append(lines)
                
                #check the length in "dep_list": 
                if(len(lines)<self.max_length_sen):
                    for i in range(self.max_length_sen-len(lines)):
                            dep_list[sentence_id].append('no_dep')
                if(len(lines)>self.max_length_sen):
                    for i in range(len(lines)-self.max_length_sen):
                            dep_list[sentence_id].pop()
                            
                sentence_id+=1
        
        #Dependencies in each sentences.
        #It's format (N*length_sen*dep_length):
        #   "N" is length of sentences in train data or test data
        #   "length_sen" is word number in each sentences
        #   "dep_length" is 20
        dep_vec=np.zeros((sen_number,self.max_length_sen,self.dep_length))    
        sentence_id=0
        for sen in dep_list:
            word_id=0
            for dep in sen:
                dep_vec[sentence_id][word_id]=self.dep_dict_init[dep]
                word_id+=1
            sentence_id+=1   
        #At last,concatenate "sen_vec3D" and "pos_vec" and "dep_vec"
        #      =>self.word_pos_vec3D
        word_pos_vec3D=np.concatenate((word_vec3D,pos_vec,dep_vec),axis=2)
#        word_pos_vec3D=np.concatenate((word_vec3D,pos_vec),axis=2)

        return word_pos_vec3D,sen_length,dep_vec
    
    def left_sdp_length(self,fore_sdp_pos_file):
        """
        Get the left SDP length
        """
        #Store the fore position
        fore_pos=[]

        with open(fore_sdp_pos_file,"r") as file:
            for line in file.readlines():
                fore_pos.append(len(line.split(" ")[:-1]))
        
        return np.float32(np.array(fore_pos))
                
    def label2id_in_data(self,label_store_filename,data_label):
        """
        In train or test data,change the traing label value to id.
        """  
        label_number=0
        with open(label_store_filename,"r") as f:
            for lines in f.readlines():
                data_label[label_number]=self.label2id[lines.strip("\r\n")]
                label_number=label_number+1
        
        return data_label
    
    def label2id_1hot(self,data_label,label2id):
        """
        Make the label in one-hot encode:[0,0,...,0,1,0,0,...,0]
        """
        onehot_encoded=[]
        for value in data_label:
            onehot=np.zeros((len(label2id)))
#            #set the Other class to "0",no pass the loss 
#            if(value!=18):
            onehot[value]=1
            onehot_encoded.append(onehot)
        return np.array(onehot_encoded)
    
    def mask_train_input(self,y_train,num_labels='all'):
        """
        2.Mask_train:mask train label.When mask=1,label can be supervised.When mask=0,label can be unsupervised.
        """
        # Construct mask_train. It has a zero where label is unknown, and one where label is known.
        if num_labels == 'all':
            # All labels are used.
            mask_train = np.ones(len(y_train), dtype=np.float32)
            print("Keeping all labels.")
        else:
            #Rough classification
            rou_num_classes=10
            # Assign labels to a subset of inputs.
            max_count = num_labels // rou_num_classes
            print("Keeping %d labels per rough class." % max_count)
            mask_train = np.zeros(len(y_train), dtype=np.float32)
            count = [0] * rou_num_classes
            for i in range(len(y_train)):
                label = y_train[i]
                rou_label=int(label)/2
                if (count[rou_label]) < max_count:
                    mask_train[i] = 1.0
                count[rou_label] += 1

        return mask_train

    def iterate_minibatches(self, inputs, targets, sen_length, left_sdp_length, batchsize, shuffle=False):
        """
        Get minibatches
        """
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt], sen_length[excerpt], left_sdp_length[excerpt]
    

    def iterate_minibatches_pi(self, inputs, targets, sen_length, batchsize, mask_train, shuffle=False):
        """
        Get minibatches in pi model train.
        """
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt], sen_length[excerpt], mask_train[excerpt]
    
    def iterate_minibatches_tempens(self, inputs, targets, sen_length, batchsize, mask_train, z_targets, shuffle=False):
        """
        Get minibatches in tempens model train.
        """
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt], sen_length[excerpt], mask_train[excerpt], z_targets[excerpt], excerpt

    def save_reverse_label(self, label_txt, label_reverse_txt):
        """
        Get the reverse label
        """
        label_reverse_save=open(label_reverse_txt,"w")
        with open(label_txt,"r") as f:
            for lines in f.readlines():
                lines=lines.strip("\r\n")
                if lines=="Other":
                    label_reverse_save.write(lines+"\n")
                else:
                    #get the first entity
                    fir_entity=lines.split("(")[1].split(",")[0]
                    #get the second entity
                    sec_entity=lines.split("(")[1].split(",")[1].split(")")[0]
                    #get the class name
                    class_name=lines.split("(")[0]
                    label_reverse_save.write(class_name+"("+sec_entity+","+fir_entity+")\n")
        label_reverse_save.close()
    
    def delete_other_in_reverse(self,training_label_reverse,training_word_pos_vec3D_reverse,training_sen_length,train_left_sdp_length_reverse):
        """
        In reverse class:delete the other class
        """
        training_word_pos_vec3D_reverse_list=[]
        training_sen_length_list=[]
        train_left_sdp_length_reverse_list=[]
        training_label_reverse_list=[]

        for i in range(len(training_label_reverse)):
            #delete other class
            if(training_label_reverse[i]!=18):
                training_word_pos_vec3D_reverse_list.append(training_word_pos_vec3D_reverse[i])
                training_sen_length_list.append(training_sen_length[i])
                train_left_sdp_length_reverse_list.append(train_left_sdp_length_reverse[i])
                training_label_reverse_list.append(training_label_reverse[i])

        return np.float32(np.array(training_word_pos_vec3D_reverse_list)),\
            np.int32(np.array(training_sen_length_list)),\
            np.int32(np.array(train_left_sdp_length_reverse_list)),\
            np.int32(np.array(training_label_reverse_list))
    
if __name__=="__main__":
    
    #1.Class:Process_data() and init the dict_word_vec
    sdp_pro=sdpProcessData()
    sdp_pro.dict_word2vec()
    sdp_pro.label2id_init()
    sdp_pro.dep_vec_init()
    
#    #2.Get word dependencies
#    #train dependencies
#    sdp_pro.word_dep(sdp_pro.train_word_dep_file,\
#        sdp_pro.train_e1_sdp_pos_file,\
#        sdp_pro.train_e2_sdp_pos_file,\
#        sdp_pro.train_sdp_link_dep_file)
#
#    #train dependencies_reverse
#    sdp_pro.word_dep(sdp_pro.train_word_dep_file,\
#        sdp_pro.train_e2_sdp_pos_file,\
#        sdp_pro.train_e1_sdp_pos_file,\
#        sdp_pro.train_sdp_link_dep_reverse_file)
#    
#    #test dependencies
#    sdp_pro.word_dep(sdp_pro.test_word_dep_file,\
#        sdp_pro.test_e1_sdp_pos_file,\
#        sdp_pro.test_e2_sdp_pos_file,\
#        sdp_pro.test_sdp_link_dep_file)
    
    #3(1).combine the two SDP in train data
    con_spd_train=sdp_pro.combine_sdp(sdp_pro.e1_sdp_train_file,sdp_pro.e2_sdp_train_file)
    con_spd_train_reverse=sdp_pro.combine_sdp(sdp_pro.e2_sdp_train_file,sdp_pro.e1_sdp_train_file)
    
    #3(2).traing_word_pos_vec3D:training data
    training_word_pos_vec3D,training_sen_length,dep_vec=sdp_pro.embedding_lookup(con_spd_train,sdp_pro.training_word_vec3D,sdp_pro.training_sen_number,sdp_pro.train_sdp_link_dep_file)
    training_word_pos_vec3D_reverse,training_sen_length,dep_vec=sdp_pro.embedding_lookup(con_spd_train_reverse,sdp_pro.training_word_vec3D_reverse,sdp_pro.training_sen_number,sdp_pro.train_sdp_link_dep_reverse_file)
    training_sen_length=np.array(training_sen_length)
    
    #3(3).left sdp length in traing data
    train_left_sdp_length=sdp_pro.left_sdp_length(sdp_pro.train_e1_sdp_pos_file)    
    train_left_sdp_length_reverse=sdp_pro.left_sdp_length(sdp_pro.train_e2_sdp_pos_file)    
    
    #4(1).combine the two SDP in test data
    con_spd_test=sdp_pro.combine_sdp(sdp_pro.e1_sdp_test_file,sdp_pro.e2_sdp_test_file)
 
    #4(2).traing_word_pos_vec3D:test data
    sdp_pro.testing_word_pos_vec3D,sdp_pro.testing_sen_length,dep_vec=sdp_pro.embedding_lookup(con_spd_test,sdp_pro.testing_word_vec3D,sdp_pro.testing_sen_number,sdp_pro.test_sdp_link_dep_file)
    testing_word_pos_vec3D=np.float32(sdp_pro.testing_word_pos_vec3D)
    testing_sen_length=np.int32(np.array(sdp_pro.testing_sen_length))
    
    #4(3).left sdp length in test data
    test_left_sdp_length=np.int32(np.reshape(sdp_pro.left_sdp_length(sdp_pro.test_e1_sdp_pos_file),(-1,1)))
    
    #5(1).training label:16000
#    # make the label reverse
#    sdp_pro.save_reverse_label(sdp_pro.train_label_store_filename,sdp_pro.train_label_reverse_txt)
    training_label=sdp_pro.label2id_in_data(sdp_pro.train_label_store_filename,\
      sdp_pro.training_label)
    training_label_reverse=sdp_pro.label2id_in_data(sdp_pro.train_label_reverse_txt,\
      sdp_pro.training_label_reverse) 
    
    #5(2).testing label:2717
    sdp_pro.testing_label=sdp_pro.label2id_in_data(sdp_pro.test_label_store_filename,\
      sdp_pro.testing_label)
    testing_label=np.int32(sdp_pro.testing_label)    
    
    #6.concatenate:
    #   training_word_pos_vec3D + training_word_pos_vec3D_reverse
    #   training_sen_length + training_sen_length_reverse
    #   train_left_sdp_length + train_left_sdp_length_reverse
    #   training_label + training_label_reverse
    training_word_pos_vec3D_reverse,\
    training_sen_length_reverse,\
    train_left_sdp_length_reverse,\
    training_label_reverse=\
        sdp_pro.delete_other_in_reverse(training_label_reverse,training_word_pos_vec3D_reverse,training_sen_length,train_left_sdp_length_reverse)
        
    training_word_pos_vec3D=np.concatenate((training_word_pos_vec3D,training_word_pos_vec3D_reverse),axis=0)
    training_word_pos_vec3D=np.float32(training_word_pos_vec3D)
    training_sen_length=np.int32(np.concatenate((training_sen_length,training_sen_length_reverse)))
    train_left_sdp_length=np.int32(np.reshape(np.concatenate((train_left_sdp_length,train_left_sdp_length_reverse),axis=0),(-1,1)))
    training_label=np.int32(np.concatenate((training_label,training_label_reverse)))
    
    #7.One-hot encode
    #label id value: Change the label to id.And 10 classes number(0-9)
    label2id=sdp_pro.label2id
    training_label_1hot=sdp_pro.label2id_1hot(training_label,label2id)
    training_label_1hot=np.int32(training_label_1hot)
    
    testing_label_1hot=sdp_pro.label2id_1hot(testing_label,label2id)    
    testing_label_1hot=np.int32(testing_label_1hot)
    