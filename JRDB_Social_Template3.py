import json
import cv2
import os
import matplotlib.pyplot as plt
import random
import numpy as np
import re
path="/home/fitadmin/PycharmProjects/pythonProject_JRDB/f_BCE+CARD+EIG+S_ACT_copy/JRDB2022/train_labels/labels_2d_stitched_new_version2/"
path_image='JRDB2022/train_images/images/image_stitched/'


def number_to_word(number):
    # Define a dictionary to map numbers to their word representations
    number_words = {
        0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
        5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine',
        10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen',
        15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen', 19: 'nineteen',
        20: 'twenty', 21: 'twenty-one', 22: 'twenty-two', 23: 'twenty-three', 24: 'twenty-four',
        25: 'twenty-five', 26: 'twenty-six', 27: 'twenty-seven', 28: 'twenty-eight', 29: 'twenty-nine',
        30: 'thirty', 31: 'thirty-one', 32: 'thirty-two', 33: 'thirty-three', 34: 'thirty-four',
        35: 'thirty-five', 36: 'thirty-six', 37: 'thirty-seven', 38: 'thirty-eight', 39: 'thirty-nine',
        40: 'forty', 41: 'forty-one', 42: 'forty-two', 43: 'forty-three', 44: 'forty-four',
        45: 'forty-five', 46: 'forty-six', 47: 'forty-seven', 48: 'forty-eight', 49: 'forty-nine',
        50: 'fifty'
    }

    # Check if the number is in the dictionary
    if number in number_words:
        return number_words[number]
    else:
        return "Number not supported"
def generate_GAE_Part(individuals_list):
    count_dict = {}
    # Iterate through the list and count occurrences
    for individual in individuals_list:
        # for individual in individuals:
            # Extract demographic information
        if len(individual)!=0:
            age = list(individual['age'].keys())[0].replace("_", " ").lower()
            gender = list(individual['gender'].keys())[0].lower()
            race = list(individual['race'].keys())[0]

            if race!='impossible' and race!='Others' :
               race=race.split('/')[1]

            # Construct a unique key for each combination of demographics
            key = f"{age} {race} {gender}"

            # Increment the count for the current combination
            count_dict[key] = count_dict.get(key, 0) + 1

    # Print the count for each combination
    list_all=[]
    for key, count in count_dict.items():
        if count>=2:
           key=key+'s'
        count=number_to_word(count)
        list_all.append(f"{count} {key}")

    return list_all
    # Example output:
    # Young_Adulthood-Female-Mongloid/Asian: 2
    # Young_Adulthood-Male-Caucasian/White: 2
    # Middle_Adulthood-Female-African/Black: 1


def check_and_convert_string(input_string):
    # List of valid words
    valid_words = {
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
        "twenty-one", "twenty-two", "twenty-three", "twenty-four", "twenty-five", "twenty-six", "twenty-seven",
        "twenty-eight", "twenty-nine", "thirty",
        "thirty-one", "thirty-two", "thirty-three", "thirty-four", "thirty-five", "thirty-six", "thirty-seven",
        "thirty-eight", "thirty-nine", "forty"
    }

    # Check if the input string is in the list of valid words
    if input_string not in valid_words:
        return "unknown"

    return input_string

def generate_sentence(demographics_infos, group_infos,social_groups,image, Cluster_ID_tagret):
    inter = 'None'
    BPC = 'None'
    location_pre = 'None'
    SSC = 'None'
    venue ='None'
    aim='None'
    sentence='None'
    verb='None'
    individual='None'
    # a dictionary
    frame=image
    group_number=0
    # data = json.load(f)
    list_person_demo=[]
    for inx,Cluster_ID in enumerate(social_groups):
        # print(len(data['labels'][image]))
        # action_label=data['labels'][image][i]["action_label"]
        # attributes= data['labels'][image][i]["attributes"]
        # box= data['labels'][image][i]["box"]
        # print(box)
        # file_id = data['labels'][image][i]["file_id"]
        # label_id = data['labels'][image][i]["label_id"]
        # # social_activity = data['labels'][image][i]["social_activity"]
        # social_group = data['labels'][image][i]["social_group"]
        # if Cluster_ID==19:
        #     print('stop')
        if Cluster_ID == Cluster_ID_tagret:
            group_number += 1
            # gender = list(data['labels'][image][i]["demographics_info"][0]['gender'].keys())[0]
            # age = list(data['labels'][image][i]["demographics_info"][0]['age'].keys())[0]
            # race = list(data['labels'][image][i]["demographics_info"][0]['race'].keys())[0]
            list_person_demo.append(demographics_infos[inx])

            if len(group_infos[inx])!=0:

                inter = list(group_infos[inx]['inter'].keys())[0].replace("&", " and ").replace("_", " ")
                BPC = list(group_infos[inx]['BPC'].keys())[0]
                location_pre = list(group_infos[inx]['location_pre'].keys())[0].replace("_", " ")
                SSC = list(group_infos[inx]['SSC'].keys())
                venue = list(group_infos[inx]['venue'].keys())[0].replace("/", " or ").replace("&", " and ").replace("_", " ")
                aim = list(group_infos[inx]['aim'].keys())[0].replace("/", " or ").replace("&", " and ").replace("_", " ")

    list_person_demo = generate_GAE_Part(list_person_demo)
    SSC=" and ".join(SSC)

    if len(list_person_demo)>=1:
        count = list_person_demo[0].lower().count("impossible")
        if count>2:
            GAE_part=''
        else:
            for i in range(len(list_person_demo)):
                list_person_demo[i]=list_person_demo[i].replace("impossible", "")
            GAE_part = " and ".join(list_person_demo)
            GAE_part=', '+GAE_part+','
            GAE_part =re.sub(r'\s{2,}', ' ', GAE_part)
    else:
        GAE_part=''
    # GAE_part =check_and_convert_string(GAE_part)
    if group_number==1:
        verb=' is '
        individual=" individual"
    elif group_number>=2:
        verb = ' are '
        individual = " individuals"

    if BPC!='unknown':
        BPC=" on the "+BPC
    else:
        BPC = ''

    if location_pre!='unknown':
        location_pre=" and "+location_pre
    else:
        location_pre = ''

    if SSC!='unknown':
        SSC=" the "+SSC
    else:
        SSC=''

    if venue!='unknown':
        venue=" in an "+ venue
    else:
        venue = ''

    if aim!='unknown':
        aim=" with the aim of "+ aim
    else:
        aim = ''

    if inter=='unknown':
        inter=""
    group_number1 = number_to_word(group_number).capitalize()
    if inter=='' and BPC=='' and location_pre=='' and SSC=='' and venue=='' and aim=='':
        sentence = f"{group_number1}{individual} is there."
    else:
        sentence = f"{group_number1}{individual}{GAE_part}{verb}{inter}{BPC}{location_pre}{SSC}{venue}{aim}."

    if group_number1 == 'Zero':
        sentence = 'The group is out of frame'
    sentence=sentence.replace("  ", " ")

    return sentence, [group_number1,individual,GAE_part,verb,inter,BPC,location_pre,SSC,venue,aim]



