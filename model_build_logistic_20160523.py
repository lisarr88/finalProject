# -*- coding: utf-8 -*-
"""
Created on Mon May 23 17:50:01 2016

@author: lisa.ryan
"""


import pandas as pd
from datetime import datetime
import calendar
import re
import numpy as np
import requests
from bs4 import BeautifulSoup
import sys
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score

shelter = pd.read_csv("C:/Users/lisa.ryan/New_Employee/General_Assembly/kaggle_shelter_animal_data/train.csv")

shelter.head()

print(shelter.columns)

# Rename missing values for name
shelter.Name.isnull().sum()  # 7,691
print(len(shelter)) # 26,729

shelter['Name'] = shelter['Name'].fillna('Unknown')
print(shelter['Name'].value_counts())

# Pull date info. Format in file: 2/12/2014  6:22:00 PM
print(type(shelter['DateTime']))
print(shelter['DateTime'].head())

shelter['DateTime'] = pd.to_datetime(shelter['DateTime'])
print(shelter['DateTime'].head())

shelter['WeekdayNum'] = shelter['DateTime'].apply(lambda x: x.weekday())
shelter['Weekday'] = shelter['DateTime'].apply(lambda x: x.strftime('%A'))


# Re-format age info
print(shelter.columns)
print(type(shelter.AgeuponOutcome))
print(shelter.AgeuponOutcome.isnull().sum())
shelter['AgeuponOutcome'] = shelter['AgeuponOutcome'].fillna('0 unknown')

age_pattern = re.compile('[^0-9]+')
age_pattern_num = re.compile('[0-9]+')
re.search(age_pattern, shelter['AgeuponOutcome'][5]).group(0)

shelter['ageString'] = shelter['AgeuponOutcome'].apply(lambda x: re.search(age_pattern, x).group(0).lower())
print(shelter['ageString'].head())
print(shelter['ageString'].value_counts())

shelter['ageNum'] = shelter['AgeuponOutcome'].apply(lambda x: int(re.search(age_pattern_num, x).group(0)))
print(shelter['ageNum'].head())

def age_fn(x):
    y = x
    if x == ' years':
        y = ' year'
    elif x == ' months':
        y = ' month'
    elif x == ' weeks':
        y = ' week'
    elif x == ' days':
        y = ' day'
    return y

def get_age_decimal(q):
    x, y = q.split('||')
    y = int(y)
    z = 0
    if x == ' year':
        z = y
    elif x == ' month':
        z = np.round(y/float(12),4)
    elif x == ' week':
        z = np.round(y/float(52),4)
    elif x == ' day':
        z = np.round(y/float(365),4)
    else:
        z = y
    return z
    
print(shelter['ageNum'][0]/12)
print(np.round(shelter['ageNum'][0]/float(12),2))

shelter['ageString'] = shelter['ageString'].apply(lambda x: age_fn(x))
print(shelter['ageString'].value_counts())

shelter['betterAge'] = shelter['ageString'].apply(lambda x: 1)
print(len(shelter['betterAge']))
print(shelter['betterAge'].head())


shelter['betterAge']=(shelter['ageString'] + '||' + shelter['ageNum'].astype(str)).apply(get_age_decimal)
print(shelter.betterAge.head(10))

print(shelter['ageString'].head(10))
print(shelter['ageNum'].head(10))


# Manipulate SexuponOutcome variable - separate Male/Female from Intact/Altered
print(shelter['SexuponOutcome'].value_counts())
print(shelter['SexuponOutcome'].isnull().sum())
shelter['SexuponOutcome'] = shelter['SexuponOutcome'].fillna('Unknown')
print(shelter['SexuponOutcome'].isnull().sum())
print(shelter['SexuponOutcome'].value_counts())

def get_gender(a):
    if a == 'Neutered Male':
        sex = 'Male'
    elif a == 'Spayed Female':
        sex = 'Female'
    elif a == 'Intact Male':
        sex = 'Male'
    elif a == 'Intact Female':
        sex = 'Female'
    else:
        sex = 'Unknown'
    return sex
    
def get_gender_type(a):
    if a == 'Neutered Male':
        sex_type = 'Altered'
    elif a == 'Spayed Female':
        sex_type = 'Altered'
    elif a == 'Intact Male':
        sex_type = 'Intact'
    elif a == 'Intact Female':
        sex_type = 'Intact'
    else:
        sex_type = 'Unknown'
    return sex_type

shelter['sex'] = shelter['SexuponOutcome'].apply(get_gender)
shelter['sex_type'] = shelter['SexuponOutcome'].apply(get_gender_type)
print(shelter['sex'].head(10))
print(shelter['sex_type'].head(10))


# Create pure bred flag
def pure_bred(z):
    has_slash = z.find("/")
    if (z[-3:] == "Mix" or z[-3:] == "mix" or has_slash > -1):
        pb = 0
    else:
        pb = 1
    return pb

shelter['pure_bred'] = shelter['Breed'].apply(pure_bred)
print(shelter['pure_bred'].value_counts())
print(shelter[shelter['pure_bred']==1]['Breed'].value_counts())





# Extract the word "Mix" from the breed variable and create an indicator variable
def clean_mix(b):
    if b[-3:] == "Mix":
        noMixBreed = b[:-4]
    elif b[-3:] == "mix":
        noMixBreed = b[:-4]
    else:
        noMixBreed = b
    return noMixBreed

shelter['betterBreed'] = shelter['Breed'].apply(clean_mix)
print(shelter['betterBreed'].head(20))

def mix_ind(c):
    if c[-3:] == "Mix":
        mixind = 1
    elif c[-3:] == "mix":
        mixind= 1
    else:
        mixind = 0
    return mixind
    
shelter['mix_ind'] = shelter['Breed'].apply(mix_ind)
print(shelter['mix_ind'].head(20))
print(shelter['mix_ind'].value_counts())


# Rename Black/Tan to make it easier to separate cross breeds from each other
def rename_hound(d):
    new_hound = d.replace("Black/Tan","Black and Tan")
    return new_hound
    
shelter['better_breed_hound'] = shelter['betterBreed'].apply(rename_hound)
print(shelter['better_breed_hound'].value_counts())


# Separate out breeds for dogs that have two breeds listed
def remove_slash1(c):
    has_slash = c.find("/")
    if has_slash > -1:
        new_str1 = c.split("/",1)[0]
    else:
        new_str1 = c
    return new_str1
    
def remove_slash2(c):
    has_slash = c.find("/")
    #return has_hound
    if has_slash > -1:
        new_str2 = c.split("/",1)[1]
    else:
        new_str2 = "Unknown"
    return new_str2

shelter['breed1'] = shelter['better_breed_hound'].apply(remove_slash1)
shelter['breed2'] = shelter['better_breed_hound'].apply(remove_slash2)
print(shelter['breed1'].value_counts())
print(shelter['breed2'].value_counts())


print(shelter['breed1'].value_counts())
print(shelter['breed2'].value_counts())

# Standardize breed names to match those listed by AKC
print(shelter['breed1'].value_counts())
print(shelter['breed2'].value_counts())

def modify_breed_names(bn):
    bnu = bn.upper()
    breed_split = bnu.split(" ",1)
    
    if breed_split[0] == 'CHIHUAHUA':
        akc_breed = 'CHIHUAHUA'
    elif bnu == 'GERMAN SHEPHERD':
        akc_breed = 'GERMAN SHEPHERD DOG'
    elif bnu == 'MINIATURE POODLE':
        akc_breed = 'POODLE'
    elif bnu == 'CATAHOULA':
        akc_breed = 'CATAHOULA LEOPARD DOG'
    elif bnu == 'AMERICAN BULLDOG':
        akc_breed = 'BULLDOG'
    elif bnu == 'ANATOL SHEPHERD':
        akc_breed = 'ANATOLIAN SHEPHERD DOG'
    elif bnu == 'STAFFORDSHIRE':
        akc_breed = 'STAFFORDSHIRE BULL TERRIER'
    elif bnu == 'PLOTT HOUND':
        akc_breed = 'PLOTT'
    elif bnu == 'AMERICAN PIT BULL TERRIER':
        akc_breed = 'PIT BULL'
    elif bnu == 'DOBERMAN PINSCH':
        akc_breed = 'DOBERMAN PINSCHER'
    elif bnu == 'QUEENSLAND HEELER':
        akc_breed = 'AUSTRALIAN CATTLE DOG'
    elif breed_split[0] == 'DACHSHUND':
        akc_breed = 'DACHSHUND'
    elif bnu == 'FLAT COAT RETRIEVER':
        akc_breed = 'FLAT-COATED RETRIEVER'
    elif bnu == 'TOY POODLE':
        akc_breed = 'POODLE'
    elif bnu == 'ENGLISH BULLDOG':
        akc_breed = 'BULLDOG'
    elif bnu == 'CHINESE SHARPEI':
        akc_breed = 'CHINESE SHAR PEI'
    elif bnu == 'COLLIE SMOOTH':
        akc_breed = 'COLLIE'
    elif bnu == 'RHOD RIDGEBACK':
        akc_breed = 'RHODESIAN RIDGEBACK'
    elif bnu == 'BRUSS GRIFFON':
        akc_breed = 'BRUSSELS GRIFFON'
    elif bnu == 'BLACK AND TAN HOUND':
        akc_breed = 'BLACK AND TAN COONHOUND'
    elif bnu == 'REDBONE HOUND':
        akc_breed = 'REDBONE COONHOUND'
    elif bnu == 'GERMAN SHORTHAIR POINTER':
        akc_breed = 'GERMAN SHORTHAIRED POINTER'
    elif bnu == 'WEST HIGHLAND TERRIER':
        akc_breed = 'WEST HIGHLAND WHITE TERRIER'
    elif bnu == 'WIRE HAIR FOX TERRIER':
        akc_breed = 'WIRE FOX TERRIER'
    elif bnu == 'STANDARD POODLE':
        akc_breed = 'POODLE'
    elif bnu == 'COLLIE ROUGH':
        akc_breed = 'COLLIE'
    elif bnu == 'ALASKAN HUSKY':
        akc_breed = 'ALASKAN MALAMUTE'
    elif bnu == 'ENGLISH POINTER':
        akc_breed = 'POINTER'
    elif bnu == 'AMERICAN ESKIMO':
        akc_breed = 'AMERICAN ESKIMO DOG'
    elif bnu == 'CHESA BAY RETR':
        akc_breed = 'CHESAPEAKE BAY RETRIEVER'
    elif bnu == 'PBGV':
        akc_breed = 'PETIT BASSET GRIFFON VENDEEN'
    elif bnu == 'ST. BERNARD SMOOTH COAT':
        akc_breed = 'ST BERNARD'
    elif bnu == 'CAVALIER SPAN':
        akc_breed = 'CAVALIER KING CHARLES SPANIEL'
    elif bnu == 'ST. BERNARD ROUGH COAT':
        akc_breed = 'ST BERNARD'
    elif bnu == 'ENGLISH COONHOUND':
        akc_breed = 'AMERICAN ENGLISH COONHOUND'
    elif bnu == 'TOY FOX TERRIER':
        akc_breed = 'SMOOTH FOX TERRIER'
    elif bnu == 'BLUETICK HOUND':
        akc_breed = 'BLUETICK COONHOUND'
    elif bnu == 'PATTERDALE TERR':
        akc_breed = 'LAKELAND TERRIER'
    elif bnu == 'LANDSEER':
        akc_breed = 'NEWFOUNDLAND'
    elif bnu == 'FEIST':
        akc_breed = 'RAT TERRIER'
    elif bnu == 'PODENGO PEQUENO':
        akc_breed = 'PORTUGUESE PODENGO PEQUENO'
    elif bnu == 'TREEING CUR':
        akc_breed = 'TREEING TENNESSEE BRINDLE'
    elif bnu == 'SCHNAUZER GIANT':
        akc_breed = 'STANDARD SCHNAUZER'
    elif bnu == 'GLEN OF IMAAL':
        akc_breed = 'GLEN OF IMAAL TERRIER'
    elif bnu == 'BOYKIN SPAN':
        akc_breed = 'BOYKIN SPANIEL'
    elif bnu == 'BULL TERRIER MINIATURE':
        akc_breed = 'BULL TERRIER'
    elif bnu == 'ENGLISH SHEPHERD':
        akc_breed = 'AUSTRALIAN SHEPHERD'
    elif bnu == 'PRESA CANARIO':
        akc_breed = 'PERRO DE PRESA CANARIO'
    elif bnu == 'PICARDY SHEEPDOG':
        akc_breed = 'BERGER PICARD'
    elif bnu == 'PORT WATER DOG':
        akc_breed = 'PORTUGUESE WATER DOG'
    elif bnu == 'SEALYHAM TERR':
        akc_breed = 'SEALYHAM TERRIER'
    elif bnu == 'ENTLEBUCHER':
        akc_breed = 'ENTLEBUCHER MOUNTAIN DOG'
    elif bnu == 'MEXICAN HAIRLESS':
        akc_breed = 'XOLOITZCUINTLI'
    else:
        akc_breed = bnu
    return akc_breed

shelter['akc_breed1'] = shelter['breed1'].apply(modify_breed_names)
print(shelter['akc_breed1'].value_counts())
shelter['akc_breed1'].value_counts().to_csv('C:/Users/lisa.ryan/New_Employee/General_Assembly/kaggle_shelter_animal_data/breed_list.csv')

shelter['akc_breed2'] = shelter['breed2'].apply(modify_breed_names)
print(shelter['akc_breed2'].value_counts())


# Transform breeds to breed groups.
# Pit bulls get their own group because they are not officially recognized as a breed by AKC.
# Breed groups were pulled from: https://en.wikipedia.org/wiki/List_of_dog_breeds_recognized_by_the_American_Kennel_Club
def add_breed_groups(breedatype):
    breed, atype = breedatype.split('||')
    if breed == 'PIT BULL':
        group = 'PIT BULL'
    elif breed == 'AFFENPINSCHER':
        group = 'TOY'
    elif breed == 'AFGHAN HOUND':
        group = 'HOUND'
    elif breed == 'AIREDALE TERRIER':
        group = 'TERRIER'
    elif breed == 'AKITA':
        group = 'WORKING'
    elif breed == 'ALASKAN MALAMUTE':
        group = 'WORKING'
    elif breed == 'AMERICAN ENGLISH COONHOUND':
        group = 'HOUND'
    elif breed == 'AMERICAN ESKIMO DOG':
        group = 'NON-SPORTING'
    elif breed == 'AMERICAN FOXHOUND':
        group = 'HOUND'
    elif breed == 'AMERICAN HAIRLESS TERRIER':
        group = 'TERRIER'
    elif breed == 'AMERICAN STAFFORDSHIRE TERRIER':
        group = 'TERRIER'
    elif breed == 'AMERICAN WATER SPANIEL':
        group = 'SPORTING'
    elif breed == 'ANATOLIAN SHEPHERD DOG':
        group = 'WORKING'
    elif breed == 'AUSTRALIAN CATTLE DOG':
        group = 'HERDING'
    elif breed == 'AUSTRALIAN SHEPHERD':
        group = 'HERDING'
    elif breed == 'AUSTRALIAN TERRIER':
        group = 'TERRIER'
    elif breed == 'BASENJI':
        group = 'HOUND'
    elif breed == 'BASSET HOUND':
        group = 'HOUND'
    elif breed == 'BEAGLE':
        group = 'HOUND'
    elif breed == 'BEARDED COLLIE':
        group = 'HERDING'
    elif breed == 'BEAUCERON':
        group = 'HERDING'
    elif breed == 'BEDLINGTON TERRIER':
        group = 'TERRIER'
    elif breed == 'BELGIAN MALINOIS':
        group = 'HERDING'
    elif breed == 'BELGIAN SHEEPDOG':
        group = 'HERDING'
    elif breed == 'BELGIAN TERVUREN':
        group = 'HERDING'
    elif breed == 'BERGAMASCO':
        group = 'HERDING'
    elif breed == 'BERGER PICARD':
        group = 'HERDING'
    elif breed == 'BERNESE MOUNTAIN DOG':
        group = 'WORKING'
    elif breed == 'BICHON FRISE':
        group = 'NON-SPORTING'
    elif breed == 'BLACK AND TAN COONHOUND':
        group = 'HOUND'
    elif breed == 'BLACK MOUTH CUR':
        group = 'HOUND'
    elif breed == 'BLACK RUSSIAN TERRIER':
        group = 'WORKING'
    elif breed == 'BLOODHOUND':
        group = 'HOUND'
    elif breed == 'BLUE LACY':
        group = 'WORKING'
    elif breed == 'BLUETICK COONHOUND':
        group = 'HOUND'
    elif breed == 'BOERBOEL':
        group = 'WORKING'
    elif breed == 'BORDER COLLIE':
        group = 'HERDING'
    elif breed == 'BORDER TERRIER':
        group = 'TERRIER'
    elif breed == 'BORZOI':
        group = 'HOUND'
    elif breed == 'BOSTON TERRIER':
        group = 'NON-SPORTING'
    elif breed == 'BOUVIER DES FLANDRES':
        group = 'HERDING'
    elif breed == 'BOXER':
        group = 'WORKING'
    elif breed == 'BOYKIN SPANIEL':
        group = 'SPORTING'
    elif breed == 'BRIARD':
        group = 'HERDING'
    elif breed == 'BRITTANY':
        group = 'SPORTING'
    elif breed == 'BRUSSELS GRIFFON':
        group = 'TOY'
    elif breed == 'BULL TERRIER':
        group = 'TERRIER'
    elif breed == 'BULL TERRIER':
        group = 'TERRIER'
    elif breed == 'BULLDOG':
        group = 'NON-SPORTING'
    elif breed == 'BULLMASTIFF':
        group = 'WORKING'
    elif breed == 'CAIRN TERRIER':
        group = 'TERRIER'
    elif breed == 'CANAAN DOG':
        group = 'WORKING'
    elif breed == 'CANE CORSO':
        group = 'WORKING'
    elif breed == 'CARDIGAN WELSH CORGI':
        group = 'HERDING'
    elif breed == 'CAROLINA DOG':
        group = 'HOUND'
    elif breed == 'CAVALIER KING CHARLES SPANIEL':
        group = 'TOY'
    elif breed == 'CESKY TERRIER':
        group = 'TERRIER'
    elif breed == 'CHESAPEAKE BAY RETRIEVER':
        group = 'SPORTING'
    elif breed == 'CHIHUAHUA':
        group = 'TOY'
    elif breed == 'CHINESE CRESTED DOG':
        group = 'TOY'
    elif breed == 'CHINESE SHAR PEI':
        group = 'NON-SPORTING'
    elif breed == 'CHINOOK':
        group = 'WORKING'
    elif breed == 'CHOW CHOW':
        group = 'NON-SPORTING'
    elif breed == 'CIRNECO DELL\'ETNA':
        group = 'HOUND'
    elif breed == 'CLUMBER SPANIEL':
        group = 'SPORTING'
    elif breed == 'COCKER SPANIEL':
        group = 'SPORTING'
    elif breed == 'COLLIE':
        group = 'HERDING'
    elif breed == 'COTON DE TULEAR':
        group = 'NON-SPORTING'
    elif breed == 'CURLY-COATED RETRIEVER':
        group = 'SPORTING'
    elif breed == 'DACHSHUND':
        group = 'HOUND'
    elif breed == 'DALMATIAN':
        group = 'NON-SPORTING'
    elif breed == 'DANDIE DINMONT TERRIER':
        group = 'TERRIER'
    elif breed == 'DOBERMAN PINSCHER':
        group = 'WORKING'
    elif breed == 'DOGUE DE BORDEAUX':
        group = 'WORKING'
    elif breed == 'ENGLISH COCKER SPANIEL':
        group = 'SPORTING'
    elif breed == 'ENGLISH FOXHOUND':
        group = 'HOUND'
    elif breed == 'ENGLISH SETTER':
        group = 'SPORTING'
    elif breed == 'ENGLISH SPRINGER SPANIEL':
        group = 'SPORTING'
    elif breed == 'ENGLISH TOY SPANIEL':
        group = 'TOY'
    elif breed == 'ENTLEBUCHER MOUNTAIN DOG':
        group = 'HERDING'
    elif breed == 'FIELD SPANIEL':
        group = 'SPORTING'
    elif breed == 'FINNISH LAPPHUND':
        group = 'HERDING'
    elif breed == 'FINNISH SPITZ':
        group = 'NON-SPORTING'
    elif breed == 'FLAT-COATED RETRIEVER':
        group = 'SPORTING'
    elif breed == 'FRENCH BULLDOG':
        group = 'NON-SPORTING'
    elif breed == 'GERMAN PINSCHER':
        group = 'WORKING'
    elif breed == 'GERMAN SHEPHERD DOG':
        group = 'HERDING'
    elif breed == 'GERMAN SHORTHAIRED POINTER':
        group = 'SPORTING'
    elif breed == 'GERMAN WIREHAIRED POINTER':
        group = 'SPORTING'
    elif breed == 'GIANT SCHNAUZER':
        group = 'WORKING'
    elif breed == 'GLEN OF IMAAL TERRIER':
        group = 'TERRIER'
    elif breed == 'GOLDEN RETRIEVER':
        group = 'SPORTING'
    elif breed == 'GORDON SETTER':
        group = 'SPORTING'
    elif breed == 'GREAT DANE':
        group = 'WORKING'
    elif breed == 'GREAT PYRENEES':
        group = 'WORKING'
    elif breed == 'GREATER SWISS MOUNTAIN DOG':
        group = 'WORKING'
    elif breed == 'GREYHOUND':
        group = 'HOUND'
    elif breed == 'HARRIER':
        group = 'HOUND'
    elif breed == 'HAVANESE':
        group = 'TOY'
    elif breed == 'IBIZAN HOUND':
        group = 'HOUND'
    elif breed == 'ICELANDIC SHEEPDOG':
        group = 'HERDING'
    elif breed == 'IRISH RED AND WHITE SETTER':
        group = 'SPORTING'
    elif breed == 'IRISH SETTER':
        group = 'SPORTING'
    elif breed == 'IRISH TERRIER':
        group = 'TERRIER'
    elif breed == 'IRISH WATER SPANIEL':
        group = 'SPORTING'
    elif breed == 'IRISH WOLFHOUND':
        group = 'HOUND'
    elif breed == 'ITALIAN GREYHOUND':
        group = 'TOY'
    elif breed == 'JAPANESE CHIN':
        group = 'TOY'
    elif breed == 'KEESHOND':
        group = 'NON-SPORTING'
    elif breed == 'KERRY BLUE TERRIER':
        group = 'TERRIER'
    elif breed == 'KOMONDOR':
        group = 'WORKING'
    elif breed == 'KUVASZ':
        group = 'WORKING'
    elif breed == 'LABRADOR RETRIEVER':
        group = 'SPORTING'
    elif breed == 'LAGOTTO ROMAGNOLO':
        group = 'SPORTING'
    elif breed == 'LAKELAND TERRIER':
        group = 'TERRIER'
    elif breed == 'LEONBERGER':
        group = 'WORKING'
    elif breed == 'LHASA APSO':
        group = 'NON-SPORTING'
    elif breed == 'LOWCHEN':
        group = 'NON-SPORTING'
    elif breed == 'MALTESE':
        group = 'TOY'
    elif breed == 'MANCHESTER TERRIER':
        group = 'TERRIER'
    elif breed == 'MASTIFF':
        group = 'WORKING'
    elif breed == 'MINIATURE AMERICAN SHEPHERD':
        group = 'HERDING'
    elif breed == 'MINIATURE BULL TERRIER':
        group = 'TERRIER'
    elif breed == 'MINIATURE PINSCHER':
        group = 'TOY'
    elif breed == 'MINIATURE SCHNAUZER':
        group = 'TERRIER'
    elif breed == 'NEAPOLITAN MASTIFF':
        group = 'WORKING'
    elif breed == 'NEWFOUNDLAND':
        group = 'WORKING'
    elif breed == 'NORFOLK TERRIER':
        group = 'TERRIER'
    elif breed == 'NORWEGIAN BUHUND':
        group = 'HERDING'
    elif breed == 'NORWEGIAN ELKHOUND':
        group = 'HOUND'
    elif breed == 'NORWEGIAN LUNDEHUND':
        group = 'NON-SPORTING'
    elif breed == 'NORWICH TERRIER':
        group = 'TERRIER'
    elif breed == 'NOVA SCOTIA DUCK-TOLLING RETRIEVER':
        group = 'SPORTING'
    elif breed == 'OLD ENGLISH SHEEPDOG':
        group = 'HERDING'
    elif breed == 'OTTERHOUND':
        group = 'HOUND'
    elif breed == 'PAPILLON':
        group = 'TOY'
    elif breed == 'PARSON RUSSELL TERRIER':
        group = 'TERRIER'
    elif breed == 'PEKINGESE':
        group = 'TOY'
    elif breed == 'PEMBROKE WELSH CORGI':
        group = 'HERDING'
    elif breed == 'PETIT BASSET GRIFFON VENDÉEN':
        group = 'HOUND'
    elif breed == 'PHARAOH HOUND':
        group = 'HOUND'
    elif breed == 'PLOTT':
        group = 'HOUND'
    elif breed == 'POINTER':
        group = 'SPORTING'
    elif breed == 'POLISH LOWLAND SHEEPDOG':
        group = 'HERDING'
    elif breed == 'POMERANIAN':
        group = 'TOY'
    elif breed == 'POODLE':
        group = 'NON-SPORTING'
    elif breed == 'PORTUGUESE PODENGO PEQUENO':
        group = 'HOUND'
    elif breed == 'PORTUGUESE WATER DOG':
        group = 'WORKING'
    elif breed == 'PUG':
        group = 'TOY'
    elif breed == 'PULI':
        group = 'HERDING'
    elif breed == 'PYRENEAN SHEPHERD':
        group = 'HERDING'
    elif breed == 'RAT TERRIER':
        group = 'TERRIER'
    elif breed == 'REDBONE COONHOUND':
        group = 'HOUND'
    elif breed == 'RHODESIAN RIDGEBACK':
        group = 'HOUND'
    elif breed == 'ROTTWEILER':
        group = 'WORKING'
    elif breed == 'RUSSELL TERRIER':
        group = 'TERRIER'
    elif breed == 'ST. BERNARD':
        group = 'WORKING'
    elif breed == 'SALUKI':
        group = 'HOUND'
    elif breed == 'SAMOYED':
        group = 'WORKING'
    elif breed == 'SCHIPPERKE':
        group = 'NON-SPORTING'
    elif breed == 'SCOTTISH DEERHOUND':
        group = 'HOUND'
    elif breed == 'SCOTTISH TERRIER':
        group = 'TERRIER'
    elif breed == 'SEALYHAM TERRIER':
        group = 'TERRIER'
    elif breed == 'SHETLAND SHEEPDOG':
        group = 'HERDING'
    elif breed == 'SHIBA INU':
        group = 'NON-SPORTING'
    elif breed == 'SHIH TZU':
        group = 'TOY'
    elif breed == 'SIBERIAN HUSKY':
        group = 'WORKING'
    elif breed == 'SILKY TERRIER':
        group = 'TOY'
    elif breed == 'SKYE TERRIER':
        group = 'TERRIER'
    elif breed == 'SLOUGHI':
        group = 'HOUND'
    elif breed == 'SMOOTH FOX TERRIER':
        group = 'TERRIER'
    elif breed == 'SOFT-COATED WHEATEN TERRIER':
        group = 'TERRIER'
    elif breed == 'SPANISH WATER DOG':
        group = 'HERDING'
    elif breed == 'SPINONE ITALIANO':
        group = 'SPORTING'
    elif breed == 'STAFFORDSHIRE BULL TERRIER':
        group = 'TERRIER'
    elif breed == 'STANDARD SCHNAUZER':
        group = 'WORKING'
    elif breed == 'SUSSEX SPANIEL':
        group = 'SPORTING'
    elif breed == 'SWEDISH VALLHUND':
        group = 'HERDING'
    elif breed == 'TIBETAN MASTIFF':
        group = 'WORKING'
    elif breed == 'TIBETAN SPANIEL':
        group = 'NON-SPORTING'
    elif breed == 'TIBETAN TERRIER':
        group = 'NON-SPORTING'
    elif breed == 'TOY FOX TERRIER':
        group = 'TOY'
    elif breed == 'TREEING WALKER COONHOUND':
        group = 'HOUND'
    elif breed == 'VIZSLA':
        group = 'SPORTING'
    elif breed == 'WEIMARANER':
        group = 'SPORTING'
    elif breed == 'WELSH SPRINGER SPANIEL':
        group = 'SPORTING'
    elif breed == 'WELSH TERRIER':
        group = 'TERRIER'
    elif breed == 'WEST HIGHLAND WHITE TERRIER':
        group = 'TERRIER'
    elif breed == 'WHIPPET':
        group = 'HOUND'
    elif breed == 'WIRE FOX TERRIER':
        group = 'TERRIER'
    elif breed == 'WIREHAIRED POINTING GRIFFON':
        group = 'SPORTING'
    elif breed == 'WIREHAIRED VIZSLA':
        group = 'SPORTING'
    elif breed == 'XOLOITZCUINTLI':
        group = 'NON-SPORTING'
    elif breed == 'YORKSHIRE TERRIER':
        group = 'TOY'
    elif breed == 'DOMESTIC SHORTHAIR' or breed == 'DOMESTIC MEDIUM HAIR' or breed == 'DOMESTIC LONGHAIR' or breed == 'SIAMESE':
        group = breed
    elif atype == 'Cat':
        group = 'OTHERCAT'
    elif atype == 'Dog':
        group = 'OTHERDOG'
    else:
        group = 'UNKNOWN'
    return group

shelter['breed_groups1']=(shelter['akc_breed1'] + '||' + shelter['AnimalType']).apply(add_breed_groups)
shelter['breed_groups1'].value_counts()

shelter['breed_groups2']=(shelter['akc_breed2'] + '||' + shelter['AnimalType']).apply(add_breed_groups)
shelter['breed_groups2'].value_counts()
 
#shelter['akc_breed1'].value_counts().to_csv('C:/Users/lisa.ryan/New_Employee/General_Assembly/kaggle_shelter_animal_data/akc_breed_freq.csv')


# Pull colors from the color field
def remove_slash_color1(e):
    has_slash = e.find("/")
    if has_slash > -1:
        new_str1, new_str2 = e.split("/",1)
    else:
        new_str1 = e
    return new_str1
    
shelter['color1'] = shelter['Color'].apply(remove_slash_color1)
print(shelter['color1'].value_counts())

def remove_slash_color2(f):
    has_slash = f.find("/")
    if has_slash > -1:
        new_str1, new_str2 = f.split("/",1)
    else:
        new_str2 = "Unknown"
    return new_str2
    
shelter['color2'] = shelter['Color'].apply(remove_slash_color2)
print(shelter['color2'].value_counts())


# Separate colors from color descriptors. For example, Black Tabby becomes two separate variables: Black and Tabby.
def remove_color_descriptions(g):
    has_slash = g.find(" ")
    if has_slash > -1:
        new_str1, new_str2 = g.split(" ",1)
    else:
        new_str1 = g
    return new_str1
    
shelter['color1_nodesc'] = shelter['color1'].apply(remove_color_descriptions)
shelter['color2_nodesc'] = shelter['color2'].apply(remove_color_descriptions)
print(shelter['color1_nodesc'].value_counts())
print(shelter['color2_nodesc'].value_counts())


def color_description_ind(h):
    has_slash = h.find(" ")
    if has_slash > -1:
        new_str2 = h.split(" ",1)[1]
    else:
        new_str2 = "Unknown"
    return new_str2
    
shelter['color1_desc_ind'] = shelter['color1'].apply(color_description_ind)
shelter['color2_desc_ind'] = shelter['color2'].apply(color_description_ind)
print(shelter['color1_desc_ind'].value_counts())
print(shelter['color2_desc_ind'].value_counts())


# Create name vs no name flag
print(shelter['Name'].value_counts())
def has_name(name):
    if name == 'Unknown':
        name_flag = 0
    else:
        name_flag = 1
    return name_flag

shelter['name_flag'] = shelter['Name'].apply(has_name)


# Quick check of dataset so far:
#shelter.to_csv('C:/Users/lisa.ryan/New_Employee/General_Assembly/kaggle_shelter_animal_data/train1.csv')


# Prep data for modeling - convert character values to numeric, etc.
animal_type = pd.get_dummies(shelter['AnimalType'])
print(animal_type.head())

sex_inds = pd.get_dummies(shelter['sex'])
print(sex_inds.head())
sex_inds = sex_inds.drop('Unknown',axis=1)
print(sex_inds.head())

sex_type_inds = pd.get_dummies(shelter['sex_type'])
print(sex_type_inds.head())
sex_type_inds = sex_type_inds.drop('Unknown',axis=1)
print(sex_type_inds.head())

color1 = pd.get_dummies(shelter['color1_nodesc'], prefix='color1')
print(color1.head())
print(color1.columns)

color2 = pd.get_dummies(shelter['color2_nodesc'], prefix='color2')
print(color2.head())
print(color2.columns)

color1_type = pd.get_dummies(shelter['color1_desc_ind'], prefix='type1')
print(color1_type.head())
print(color1_type.columns)
color1_type = color1_type.drop('type1_Unknown', axis=1)
print(color1_type.head())
print(color1_type.columns)

color2_type = pd.get_dummies(shelter['color2_desc_ind'], prefix='type2')
print(color2_type.head())
print(color2_type.columns)
color2_type = color2_type.drop('type2_Unknown', axis=1)
print(color2_type.head())
print(color2_type.columns)

breed_groups1 = pd.get_dummies(shelter['breed_groups1'], prefix='group1')
print(breed_groups1.head())
print(breed_groups1.columns)

breed_groups2 = pd.get_dummies(shelter['breed_groups2'], prefix='group2')
print(breed_groups2.head())
print(breed_groups2.columns)

akc_breed1 = pd.get_dummies(shelter['akc_breed1'], prefix='breed1')
print(akc_breed1.head())
print(akc_breed1.columns)

akc_breed2 = pd.get_dummies(shelter['akc_breed2'], prefix='breed2')
print(akc_breed2.head())
print(akc_breed2.columns)


print(shelter.columns)
model_build = shelter
print(model_build.columns)
model_build = model_build.drop('AnimalID',axis=1)
model_build = model_build.drop('Name',axis=1)
model_build = model_build.drop('DateTime',axis=1)
model_build = model_build.drop('OutcomeSubtype',axis=1)
model_build = model_build.drop('AnimalType',axis=1)
model_build = model_build.drop('SexuponOutcome',axis=1)
model_build = model_build.drop('AgeuponOutcome',axis=1)
model_build = model_build.drop('Breed',axis=1)
model_build = model_build.drop('Color',axis=1)
model_build = model_build.drop('ageString',axis=1)
model_build = model_build.drop('ageNum',axis=1)
model_build = model_build.drop('sex',axis=1)
model_build = model_build.drop('sex_type',axis=1)
model_build = model_build.drop('betterBreed',axis=1)
model_build = model_build.drop('better_breed_hound',axis=1)
model_build = model_build.drop('breed1',axis=1)
model_build = model_build.drop('breed2',axis=1)
model_build = model_build.drop('color1',axis=1)
model_build = model_build.drop('color2',axis=1)
model_build = model_build.drop('color1_nodesc',axis=1)
model_build = model_build.drop('color2_nodesc',axis=1)
model_build = model_build.drop('color1_desc_ind',axis=1)
model_build = model_build.drop('color2_desc_ind',axis=1)
model_build = model_build.drop('akc_breed1',axis=1)
model_build = model_build.drop('akc_breed2',axis=1)
model_build = model_build.drop('breed_groups1',axis=1)
model_build = model_build.drop('breed_groups2',axis=1)
model_build = model_build.drop('WeekdayNum',axis=1)
model_build = model_build.drop('Weekday', axis=1)


print(model_build.columns)

print(model_build['OutcomeType'].value_counts())

def outcome_to_num(out):
    if out == 'Adoption':
        outcome = 1
    elif out == 'Transfer':
        outcome = 2
    elif out == 'Return_to_owner':
        outcome = 3
    elif out == 'Euthanasia':
        outcome = 4
    elif out == 'Died':
        outcome = 5
    return outcome

model_build['outcome'] = model_build['OutcomeType'].apply(outcome_to_num)
print(model_build['outcome'].value_counts())

model_build = model_build.drop('OutcomeType',axis=1)
model_build.columns

model_build1 = pd.concat([model_build, animal_type, sex_inds, color1, color1_type, breed_groups1], axis = 1) # 0.80758467
model_build1 = model_build1.drop('color1_Ruddy',axis=1)


model_build1.columns
print(model_build1.head(10))

print(model_build1.head(10))
print(model_build1.columns)





# Logistic Regression Model
model_build1.columns
model_build2 = model_build1.drop('outcome',axis=1)
model_build2.columns
feat_col = model_build2.columns
print(feat_col)
print(len(feat_col))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(model_build2, model_build1.outcome, test_size = 0.35)

print(X_train)
print(X_test)
print(y_train)
print(y_test)

X_train1 = X_train.as_matrix(X_train.columns)
X_test1 = X_test.as_matrix(X_test.columns)
y_train1 = y_train.as_matrix().ravel()
y_test1 = y_test.as_matrix().ravel()

print(len(X_train1))
print(len(X_test1))
print(len(y_train1))
print(len(y_test1))


logreg = LogisticRegression(C=1e9, multi_class='ovr')

logreg_fit = logreg.fit(X_train1, y_train1)



lr_predict_proba = logreg_fit.predict_proba(X_test1)
log_loss(y_test1,lr_predict_proba,eps=1e-15)
print(lr_predict_proba)
print(y_test1)

lr_predict = logreg_fit.predict(X_test1)
print metrics.confusion_matrix(y_test1, lr_predict)
print(len(y_test1))


accuracy = logreg_fit.score(X_test1, y_test1)
print(accuracy)






# accuracy:
accuracy = logreg_fit.score(X_test1, y_test1)
print(accuracy)

# sensitivity:
conf_matrix = metrics.confusion_matrix(y_test1, lr_predict)
conf_matrix
print(conf_matrix[0,0]/float(conf_matrix[0,0]+conf_matrix[0,1]+conf_matrix[0,2]+conf_matrix[0,3]+conf_matrix[0,4]))
print(conf_matrix[1,1]/float(conf_matrix[1,1]+conf_matrix[1,0]+conf_matrix[1,2]+conf_matrix[1,3]+conf_matrix[1,4]))
print(conf_matrix[2,2]/float(conf_matrix[2,2]+conf_matrix[2,0]+conf_matrix[2,1]+conf_matrix[2,3]+conf_matrix[2,4]))
print(conf_matrix[3,3]/float(conf_matrix[3,3]+conf_matrix[3,0]+conf_matrix[3,1]+conf_matrix[3,2]+conf_matrix[3,4]))
print(conf_matrix[4,4]/float(conf_matrix[4,4]+conf_matrix[4,0]+conf_matrix[4,1]+conf_matrix[4,2]+conf_matrix[4,3]))

# specificity:
conf_matrix = metrics.confusion_matrix(y_test1, lr_predict)
conf_matrix
TN1 = (conf_matrix[1,1]+conf_matrix[1,2]+conf_matrix[1,3]+conf_matrix[1,4]+conf_matrix[2,1]+conf_matrix[2,2]+conf_matrix[2,3]+conf_matrix[2,4]+conf_matrix[3,1]+conf_matrix[3,2]+conf_matrix[3,3]+conf_matrix[3,4]+conf_matrix[4,1]+conf_matrix[4,2]+conf_matrix[4,3]+conf_matrix[4,4])
FP1 = (conf_matrix[1,0]+conf_matrix[2,0]+conf_matrix[3,0]+conf_matrix[4,0])
TN2 = (conf_matrix[0,0]+conf_matrix[0,2]+conf_matrix[0,3]+conf_matrix[0,4]+conf_matrix[2,0]+conf_matrix[2,2]+conf_matrix[2,3]+conf_matrix[2,4]+conf_matrix[3,0]+conf_matrix[3,2]+conf_matrix[3,3]+conf_matrix[3,4]+conf_matrix[4,0]+conf_matrix[4,2]+conf_matrix[4,3]+conf_matrix[4,4])
FP2 = (conf_matrix[0,1]+conf_matrix[2,1]+conf_matrix[3,1]+conf_matrix[4,1])
TN3 = (conf_matrix[0,0]+conf_matrix[0,1]+conf_matrix[0,3]+conf_matrix[0,4]+conf_matrix[1,0]+conf_matrix[1,1]+conf_matrix[1,3]+conf_matrix[1,4]+conf_matrix[3,0]+conf_matrix[3,1]+conf_matrix[3,3]+conf_matrix[3,4]+conf_matrix[4,0]+conf_matrix[4,1]+conf_matrix[4,3]+conf_matrix[4,4])
FP3 = (conf_matrix[0,2]+conf_matrix[1,2]+conf_matrix[3,2]+conf_matrix[4,2])
TN4 = (conf_matrix[0,0]+conf_matrix[0,1]+conf_matrix[0,2]+conf_matrix[0,4]+conf_matrix[1,0]+conf_matrix[1,1]+conf_matrix[1,2]+conf_matrix[1,4]+conf_matrix[2,0]+conf_matrix[2,1]+conf_matrix[2,2]+conf_matrix[2,4]+conf_matrix[4,0]+conf_matrix[4,1]+conf_matrix[4,2]+conf_matrix[4,4])
FP4 = (conf_matrix[0,3]+conf_matrix[1,3]+conf_matrix[2,3]+conf_matrix[4,3])
TN5 = (conf_matrix[0,0]+conf_matrix[0,1]+conf_matrix[0,2]+conf_matrix[0,3]+conf_matrix[1,0]+conf_matrix[1,1]+conf_matrix[1,2]+conf_matrix[1,3]+conf_matrix[2,0]+conf_matrix[2,1]+conf_matrix[2,2]+conf_matrix[2,3]+conf_matrix[3,0]+conf_matrix[3,1]+conf_matrix[3,2]+conf_matrix[3,3])
FP5 = (conf_matrix[0,4]+conf_matrix[1,4]+conf_matrix[2,4]+conf_matrix[3,4])


print(TN1/float(TN1+FP1))
print(TN2/float(TN2+FP2))
print(TN3/float(TN3+FP3))
print(TN4/float(TN4+FP4))
print(TN5/float(TN5+FP5))


lr_predict_proba = logreg_fit.predict_proba(X_test1)
print(lr_predict_proba)
print(y_test1)

log_loss(y_test1,lr_predict_proba,eps=1e-15)

crossval = cross_val_score(logreg,X_train,y_train,scoring="accuracy",cv=5)
crossval











# logistic regression with 2 outcome classes
def outcome_to_num(out):
    if out == 'Adoption' or out == 'Transfer' or out == 'Return_to_owner':
        outcome = 0
    elif out == 'Euthanasia' or out == 'Died':
        outcome = 1
    return outcome

model_build['outcome'] = model_build['OutcomeType'].apply(outcome_to_num)
print(model_build['outcome'].value_counts())

model_build = model_build.drop('OutcomeType',axis=1)
model_build.columns

model_build1 = pd.concat([model_build, animal_type, sex_inds, color1, color1_type, breed_groups1], axis = 1) # 0.80758467
model_build1 = model_build1.drop('color1_Ruddy',axis=1)


model_build1.columns
print(model_build1.head(10))

print(model_build1.head(10))
print(model_build1.columns)



model_build1.columns
model_build2 = model_build1.drop('outcome',axis=1)
model_build2.columns
feat_col = model_build2.columns
print(feat_col)
print(len(feat_col))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(model_build2, model_build1.outcome, test_size = 0.35)

print(X_train)
print(X_test)
print(y_train)
print(y_test)

X_train1 = X_train.as_matrix(X_train.columns)
X_test1 = X_test.as_matrix(X_test.columns)
y_train1 = y_train.as_matrix().ravel()
y_test1 = y_test.as_matrix().ravel()

print(len(X_train1))
print(len(X_test1))
print(len(y_train1))
print(len(y_test1))



logreg = LogisticRegression(C=1e9)

logreg_fit = logreg.fit(X_train1, y_train1)



lr_predict_proba = logreg_fit.predict_proba(X_test1)
log_loss(y_test1,lr_predict_proba,eps=1e-15)
print(lr_predict_proba)
print(y_test1)

lr_predict = logreg_fit.predict(X_test1)
print metrics.confusion_matrix(y_test1, lr_predict)
print(len(y_test1))


accuracy = logreg_fit.score(X_test1, y_test1)
print(accuracy)



