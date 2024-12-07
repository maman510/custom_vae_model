import datetime
import pickle
import datetime as dt
import os 
# from bs4 import BeautifulSoup
# import lxml
import keras.datasets
import keras.datasets.cifar100
import requests
import random
from inspect import getmembers
import json
import time
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf

def get_current_timestamp():
    dt_obj = datetime.datetime.now()
    current_time = dt_obj.strftime('%I:%M:%S %p')
    timestamp = f"{Get_String_Date()} {current_time}"
    return timestamp


def print_members(obj, obj_name="placeholder_name"):
    """Print members of given COM object"""
    try:
        fields = list(obj._prop_map_get_.keys())
    except AttributeError:
        print("Object has no attribute '_prop_map_get_'")
        print("Check if the initial COM object was created with"
              "'win32com.client.gencache.EnsureDispatch()'")
        raise
    methods = [m[0] for m in getmembers(obj) if (not m[0].startswith("_")
                                                 and "clsid" not in m[0].lower())]

    if len(fields) + len(methods) > 0:
        print("Members of '{}' ({}):".format(obj_name, obj))
    else:
        raise ValueError("Object has no members to print")

    print("\tFields:")
    if fields:
        for field in fields:
            print(f"\t\t{field}")
    else:
        print("\t\tObject has no fields to print")

    print("\tMethods:")
    if methods:
        for method in methods:
            print(f"\t\t{method}")
    else:
        print("\t\tObject has no methods to print")


def Create_Pkl(path,data):
    
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

def Load_Pkl(path):
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def Get_String_Date(date=None):
    if date == None:
        dt = datetime.datetime.now()
        return str(dt).split(" ")[0]             

    else:
        #implement any date later
        return 0

def File_Checker(file_path):
    file_exists = os.path.isfile(file_path)
    return file_exists


def Iter(some_list):
    container = []
    
    for i in range(len(some_list)):
        print(some_list[i])
        container.append(some_list[i])
    
    return container

def Get_Soup(some_url):
    req = requests.get(some_url).content
    soup = BeautifulSoup(req,'lxml')
    return soup 

def get_color_options():
        colors = {
        "Black": "\033[0;30m",
        "Dark Gray": "\033[1;30m", 
        "Red": "\033[0;31m",
        "Light Red": "\033[1;31m",
        "Green": "\033[0;32m",
        "Light Green": "\033[1;32m",
        "Orange": "\033[0;33m",
        "Yellow": "\033[1;33m",
        "Blue": "\033[0;34m",
        "Light Blue": "\033[1;34m",
        "Purple": "\033[0;35m",
        "Light Purple": "\033[1;35m",
        "Cyan": "\033[0;36m",
        "Light Cyan": "\033[1;36m",
        "Light Gray": "\033[0;37m",
        "White": "\033[1;37m",
    }
        return [k for k in colors.keys()]
def color_string(some_color,some_string):

    colors = {
        "Black": "\033[0;30m",
        "Dark Gray": "\033[1;30m", 
        "Red": "\033[0;31m",
        "Light Red": "\033[1;31m",
        "Green": "\033[0;32m",
        "Light Green": "\033[1;32m",
        "Orange": "\033[0;33m",
        "Yellow": "\033[1;33m",
        "Blue": "\033[0;34m",
        "Light Blue": "\033[1;34m",
        "Purple": "\033[0;35m",
        "Light Purple": "\033[1;35m",
        "Cyan": "\033[0;36m",
        "Light Cyan": "\033[1;36m",
        "Light Gray": "\033[0;37m",
        "White": "\033[1;37m",
        "None": '\033[0m'
    }
    s = f"{colors[some_color.title()]} {some_string} {colors['None']}"
    
    return s

def make_request_header():

    def get_user_agent():
        user_agent_list = [
    #Chrome
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
        'Mozilla/5.0 (Windows NT 5.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
        #Firefox
        'Mozilla/4.0 (compatible; MSIE 9.0; Windows NT 6.1)',
        'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)',
        'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (Windows NT 6.2; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.0; Trident/5.0)',
        'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)',
        'Mozilla/5.0 (Windows NT 6.1; Win64; x64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)',
        'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)',
        'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0; .NET CLR 2.0.50727; .NET CLR 3.0.4506.2152; .NET CLR 3.5.30729)'
        ]
        
        return random.choice(user_agent_list)

    user = get_user_agent()             #random user-agent
    headers = {
        "User-Agent":user,
        "accept": "text/javascript, application/javascript, */*",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "en-US,en;q=0.9",
        "cache-control": "no-cache"
    }
    
    return headers


def write_json(data, path):
    with open(path, 'w') as fp:
        json.dump(data, fp,indent=4)


def read_json(path):
    with open(path, "r") as f:
        d = json.load(f)
        return d
    

def string_to_datetime(string):
    split_str = string.split("-")
    year = int(split_str[0])
    month = int(split_str[1])
    day = int(split_str[2])
    date = datetime.datetime(year, month, day).date()
    return date

def thread_process(f):
    def inner(*args):
        print("running")
        f(*args)
        return None
    return inner



def grouper(arr, g_len,padding=0):
    '''
        Returns passed flat array into a nested array with groups of len g_len;
        if len of arr not divisible by g_len, value passed as padding used to fill last group
    '''
    if type(arr[0]) == list:
        arr = [val for row in arr for val in row]
    new_arr = []
    i = 0
    nested = []
    #add remaining values from arr + padding
    
    while True:
        if len(arr) == 0:
            new_arr.append(nested)
            break
        if i == g_len:
       
            new_arr.append(nested)
       
            nested = []
            i = 0
            continue
        else:
            nested.append(arr[0])
            arr = arr[1:]
            i += 1


    #add padded group to new_arr
    if len(new_arr[-1]) != g_len:
        missing = g_len - len(new_arr[-1])
        
        last_arr = new_arr.pop() + [padding for i in range(missing)]
        last_arr = last_arr 
       
        new_arr.append(last_arr)


    return new_arr

def dot_product(m,n):
    if len(n) >= len(m):
        longer = n
        shorter = m
    else:
        longer = m
        shorter = n

    longer = [[longer[j][i] for j in range(len(longer))] for i in range(len(longer[0]))]
    results = []
    arrs = []
    for row in shorter:
        for row2 in longer:
            arrs.append([row, row2])
            results.append([row[i]*row2[i] for i in range(len(row))])


    evens = [sum(results[i]) for i in range(len(results)) if i%2 == 0]
    odds = [sum(results[i]) for i in range(len(results)) if i%2 != 0]
    matrix = evens + odds
  
    return grouper([sum(i) for i in results],len(shorter))



def factorial(n):
    if len(n) == 1:
        return n.pop()
    else:
        return  n[0] * factorial(n[1:])


def combo(n,r):
    numerator = [n-i for i in range((n - (n-r)))]
    denom = [i+1 for i in range(r)]
    return factorial(numerator)/factorial(denom)
    
def perm(n,r):
    return factorial([i+1 for i in range(n-r, n)])





def generate_prob_matrix(matrices, f=lambda m1,m2: m1 - m2):
    '''
        pass list of matrices and returns a list of probability matrices that represent probability of each cell increasing in subsequent matrices;

        ** f is a function you can pass to substitute the operation performed between matrices if you want to keep track of another stat instead of value increases
            ** THIS DOES NOT WORK YET AND SHOULD ONLY USE DEFAULT FOR NOW BECAUSE PROPER CHECK ON LINE 274 NEEDS TO BE IMPLEMENTED THAT IS IN LINE WITH function passed as f
    '''
    probability_matrices = []
    for i in range(len(matrices)):
        #start matrix; start at 0 end at -1
        start = matrices[i]

        #subtract values in start with each matrix; if value is less than 0, no increase, else, increas
        diffs = [f(start,matrix) for matrix in matrices[1:]]

        #create counter df and initiate each cell with 0 count; increment by 1 for each df in remainder of matrices (i.e. diffs) where value increases
        counts = [[0 for i in range(len(start.columns))] for j in range(len(start.index))]
        for j in range(len(diffs)):

            #mask diffs with truth value of whether it increased or decreased; if difference between values < 1, start_val < next_val
            probability = [diffs[j].loc[i:,:].values > 0 for i in start.index]
    
            #increment counts for each cell that has a TRUE value, representing an increase, leave count if else
            counts = [[counts[i][j] + 1 if probability[0][i][j] == True else counts[i][j] for j in range(len(counts[i]))] for i in range(len(counts))]

        #divide each cell in counts by number of matrices iterated through to find probability of increase per subsequent matrix
        probability_matrix = pd.DataFrame(counts)/len(diffs)
        probability_matrices.append(probability_matrix)
    
    return probability_matrices




def get_wiki_images(query):
    import wikipedia
    import os
    import tool_box



    cwd = os.getcwd()
    new_dir_path = f"{cwd}/{'_'.join(query.split(' '))}_images"
    if os.path.exists(new_dir_path) == False:
        os.mkdir(new_dir_path)
    else:
        q = input(f"\nPATH " + tool_box.color_string("yellow", f"{new_dir_path}") +  "ALREADY EXIST; " + tool_box.color_string("red",f"OVERWRITE? ('Y' or 'N')\n"))
        if q.upper() == "Y":


            image_queries = wikipedia.search(query)
            image_urls = []
            total = 0
            for i in image_queries:
                print(i)
                try:
                    urls = wikipedia.page(i.lower()).images
                    image_urls.extend([url for url in urls if url.split(".")[-1] != "svg" and url.split(".")[-1] != "png"])
                    total += len(image_urls)
                    print(total)
                except:
                    pass

          
            os.chdir(new_dir_path)
            for i in range(len(image_urls)):

                cmd = f"curl -O {image_urls[i]}"
                os.system(cmd)
        else:
            return None
            
            

def camelcase(string):
    split_string = string.split(" ")
    return ''.join([split_string[i].lower() if i == 0 else split_string[i].title() for i in range(len(split_string))])



def read_txt(path):

    with open(path, "r") as f:
        return f.read()
    


def timer(f):
    def nested():
        start = time.time()
        f()
        print(f"{color_string('yellow',f.__name__)} TOOK: {color_string('yellow',time.time() - start)} to complete")
         
    return nested

def get_federal_holidays():
        
    dr = pd.date_range(start='1990-01-01', end=get_current_timestamp())
    df = pd.DataFrame()
    df['Date'] = dr
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=dr.min(), end=dr.max())

    return holidays


def plot_image(image):
    '''
        accepts image path or ndarray representation of image for plotting
    '''
    if type(image) == str:
        # Load an image if path passed
        image = mpimg.imread(image)


    plt.imshow(image.squeeze())
    plt.axis('off')  # Turn off axis labels
    plt.show()




def load_tf_dataset(dataset_name, test_split=0.2, image_reshape_size=(32,32,3)):
        
    # Load the Flowers dataset (with 5 categories)
    dataset, info = tfds.load(dataset_name, with_info=True, as_supervised=True)

    # The dataset is split into train and test
    all_data = dataset['train']

    #split test using ratio
    test_start_index = int(len(all_data) * (1.0 - test_split)) # leaves first 100% - split_pct for train and remaining for test (ex: split=0.2 => 100% - 20% = 80% of the elements left out from beginning for train)

    # Convert to NumPy ndarrays
    train_images = []
    train_labels = []

    test_images = []
    test_labels = []

    # # Iterate through the train dataset
    count = 0
    current_image_container = []
    current_label_container = []

    #iterate through all_data until train data satisfied - fill remaining and assign to test
    print(color_string("yellow", f"\n\n\tLOADING DATASET (shape = {image_reshape_size})....\n\n"))
    for image, label in all_data:

        if count == test_start_index:
            #set train to current container and reset current container for test to use
            train_images = np.array([image for image in current_image_container])
            train_labels = np.array([image for image in current_label_container])
            current_image_container = test_images
            current_label_container = test_labels

        # Reshape the image to 32x32x3
       # print(count, test_start_index, len(train_images))
        
        image_resized = tf.image.resize(image, [image_reshape_size[0], image_reshape_size[1]]).numpy()

        current_image_container.append(image_resized)
        current_label_container.append(label.numpy())
        count += 1

    #set current containers for test
    test_images = np.array(current_image_container)
    test_labels = np.array(current_label_container)


    #normalize image data
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    print(f"\n\nTRAIN IMAGE COUNT: {len(train_images)}\tTEST IMAGE COUNT: {len(test_images)}\tTOTAL IN SET: {len(all_data)}")

    print(f"Train Images shape: {train_images.shape}")
    print(f"Train Labels shape: {train_labels.shape}")
    print(f"Test Images shape: {test_images.shape}")
    print(f"Test Labels shape: {test_labels.shape}")

    return (train_images, train_labels), (test_images, test_labels)

