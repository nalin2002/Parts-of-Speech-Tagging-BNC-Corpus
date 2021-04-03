import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import os
import re
from sklearn.metrics import confusion matrix

#-------------------------------------------------------------------------------------------------------------------------

directory1= r"/home/daddy_yankee/Desktop/csn371/csn371_assignment/Train-corpus"
directory2=r"/home/daddy_yankee/Desktop/csn371/csn371_assignment/Test-corpus"

#-------------------------------------------------------------------------------------------------------------------------

train_word_dictionary=dict()
train_tag_dictionary=dict()
train_word_tag_dictionary = dict()
train_tag_list= list()
train_word_list=list()
train_all_tags= list()

#-------------------------------------------------------------------------------------------------------------------------

def Prob_T2_G_T1(t2,t1,var1=train_tag_list,var2=train_tag_dictionary):
    count_t1=var2[t1]
    count_t2_and_t1=0
    for i in range(len(var1)-1):
        if var1[i]==t1 and var1[i+1]==t2 :
            count_t2_and_t1+=1
    x=(count_t2_and_t1/count_t1)
    return x

#-------------------------------------------------------------------------------------------------------------------------

def Prob_W_G_T(word, tag,train_tag_dictionary,train_word_tag_dictionary):
    string= str(word)+"_"+str(tag)
    try:
     count= train_word_tag_dictionary[string]
     x=count/(train_tag_dictionary[tag])
     return x
    except:
     return 0

#-------------------------------------------------------------------------------------------------------------------------

def Modified_Viterbi(test_words, all_dist_tags,var1=train_word_dictionary):
    final_assgn_tags=[]
    T=all_dist_tags
    likelihood=dict()
    for key,word in enumerate(test_words):
        likelihood[key]=dict()
        probs=[]

        for tag in T:
            if word in var1.keys():
                emission_prob=Prob_W_G_T(word,tag,train_tag_dictionary,train_word_tag_dictionary)
            else:
                emission_prob=0

            if key==0:
                transition_prob=transition_probabilities.loc['PUN',tag]
                likelihood[key][tag]=emission_prob*transition_prob
                probs.append(likelihood[key][tag])
            else:
                trans_states=[]
                for prev_tags in all_dist_tags:
                    transition_prob=transition_probabilities.loc[prev_tags,tag]
                    tag_likelihood=transition_prob*emission_prob*likelihood[key-1][prev_tags]
                    trans_states.append(tag_likelihood)
                max_tag_prob=max(trans_states)
                likelihood[key][tag]=max_tag_prob
                probs.append(likelihood[key][tag])

        max_prob_value=max(probs)
        max_prob_value_index=probs.index(max_prob_value)
        final_assgn_tags.append(all_dist_tags[max_prob_value_index])
    return list(final_assgn_tags)

#-------------------------------------------Training Data-----------------------------------------------------------------

for file in os.listdir(directory1):
    subfile = os.path.join(directory1, file)
    for xml_files in os.listdir(subfile):
        xml_file = os.path.abspath(os.path.join(subfile, xml_files))
        myTree = ET.parse(xml_file)
        myRoot = myTree.getroot()
        for x in myRoot.findall('.//w'):
            text=(x.text).strip().lower()
            tag=x.attrib['c5']
            tag=re.split('-',tag)[0]

            train_tag_list.append(tag)
            train_word_list.append(text)

            if tag in train_tag_dictionary:
                train_tag_dictionary[tag]+=1
            else:
                train_tag_dictionary[tag]=1

            if text in train_word_dictionary:
                train_word_dictionary[text] += 1
            else:
                train_word_dictionary[text] = 1

            string = str(text)+'_'+ str(tag)

            if string in train_word_tag_dictionary:
                train_word_tag_dictionary[string]+=1
            else:
                train_word_tag_dictionary[string]=1

        for x in myRoot.findall('.//c'):
            text = (x.text).strip().lower()
            tag = x.attrib['c5']
            tag = re.split('-', tag)[0]

            train_tag_list.append(tag)
            train_word_list.append(text)

            if tag in train_tag_dictionary:
                train_tag_dictionary[tag] += 1
            else:
                train_tag_dictionary[tag] = 1

            if text in train_word_dictionary:
                train_word_dictionary[text] += 1
            else:
                train_word_dictionary[text] = 1

            string = str(text) + '_' + str(tag)

            if string in train_word_tag_dictionary:
                train_word_tag_dictionary[string] += 1
            else:
                train_word_tag_dictionary[string] = 1

train_dist_all_tags=list(train_tag_dictionary.keys())
train_dist_all_words=list(train_word_dictionary.keys())

#---------------------------------------Transition Matrix-----------------------------------------------------------------

tp=np.zeros((len(train_dist_all_tags),len(train_dist_all_tags)),dtype='float32')
for i,tag1 in enumerate(train_dist_all_tags):
    for j,tag2 in enumerate(train_dist_all_tags):
        tp[i,j]=Prob_T2_G_T1(tag2,tag1)

transition_probabilities=pd.DataFrame(tp, columns=train_dist_all_tags, index= train_dist_all_tags)

#---------------------------------------Emission Matrix-------------------------------------------------------------------

cp = np.zeros((len(train_dist_all_tags), len(train_dist_all_words)),dtype='float32')
for i, tag in enumerate(train_dist_all_tags):
    for j, word in enumerate(train_dist_all_words):
        cp[i, j] = Prob_W_G_T(word, tag,train_tag_dictionary,train_word_tag_dictionary)

condition_probabilities = pd.DataFrame(cp, columns=train_dist_all_words, index=train_dist_all_tags)

#----------------------------------------Testing Data---------------------------------------------------------------------

test_tags=dict()
test_words=dict()
test_tags_list=list()
test_words_list=list()
Pred_test=list()
for file in os.listdir(directory2):
    subfile = os.path.join(directory2, file)
    for xml_files in os.listdir(subfile):
        xml_file = os.path.abspath(os.path.join(subfile, xml_files))
        myTree = ET.parse(xml_file)
        myRoot = myTree.getroot()
        for x in myRoot.findall('.//w'):
            text=(x.text).strip().lower()
            w= re.split('-',x.attrib['c5'])[0]

            if text in test_words:
                test_words[text]+=1
            else:
                test_words[text]=1

            if w in test_tags:
                test_tags[w]+=1
            else:
                test_tags[w]=1

            test_tags_list.append(w)
            test_words_list.append(text)
        for x in myRoot.findall('.//c'):
            text = (x.text).strip().lower()
            w = re.split('-', x.attrib['c5'])[0]

            if text in test_words:
                test_words[text] += 1
            else:
                test_words[text] = 1

            if w in test_tags:
                test_tags[w] += 1
            else:
                test_tags[w] = 1

            test_tags_list.append(w)
            test_words_list.append(text)
        predicted_tags=Modified_Viterbi(test_words_list,train_dist_all_tags)
        Pred_test=list(predicted_tags)

#---------------------------------------------Accuracy-------------------------------------------------------------------

Actual_test=test_tags_list
numerator=0
denominator=len(Actual_test)
for i in range(len(Actual_test)):
    if Actual_test[i]==Pred_test[i]:
        numerator+=1
x=(numerator/denominator)
x=x*100
print("Accuracy = "+str(x))

#------------------------------------------Confusion Matrix--------------------------------------------------------------

print("Confusion Matrix = ")
print("\n")
print(confusion_matrix(Actual_test,Pred_test))
print("\n")