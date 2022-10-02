import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# MotionSense Preprocessing
def get_ds_infos(path):
    """
    Read the file includes data subject information.

    Data Columns:
    0: code [1-24]
    1: weight [kg]
    2: height [cm]
    3: age [years]
    4: gender [0:Female, 1:Male]

    Returns:
        A pandas DataFrame that contains inforamtion about data subjects' attributes
    """

    dss = pd.read_csv(path + "data_subjects_info.csv")
    print("[INFO] -- Data subjects' information is imported.")

    return dss


def set_data_types(data_types=["userAcceleration"]):
    """
    Select the sensors and the mode to shape the final dataset.

    Args:
        data_types: A list of sensor data type from this list: [attitude, gravity, rotationRate, userAcceleration]
    Returns:
        It returns a list of columns to use for creating time-series from files.
    """
    dt_list = []
    for t in data_types:
        if t != "attitude":
            dt_list.append([t + ".x", t + ".y", t + ".z"])
        else:
            dt_list.append([t + ".roll", t + ".pitch", t + ".yaw"])

    return dt_list


def creat_time_series(path, dt_list, act_labels, trial_codes, mode="mag", labeled=True, combine_grav_acc=False):
    """
    Args:
        dt_list: A list of columns that shows the type of data we want.
        act_labels: list of activites
        trial_codes: list of trials
        mode: It can be "raw" which means you want raw data
        for every dimention of each data type,
        [attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)].
        or it can be "mag" which means you only want the magnitude for each data type: (x^2+y^2+z^2)^(1/2)
        labeled: True, if we want a labeld dataset. False, if we only want sensor values.
        combine_grav_acc: True, means adding each axis of gravity to  corresponding axis of userAcceleration.
    Returns:
        It returns a time-series of sensor data.

    """
    num_data_cols = len(dt_list) if mode == "mag" else len(dt_list * 3)

    if labeled:
        dataset = np.zeros((0, num_data_cols + 7))  # "7" --> [act, code, weight, height, age, gender, trial]
    else:
        dataset = np.zeros((0, num_data_cols))

    ds_list = get_ds_infos(path)

    print("[INFO] -- Creating Time-Series")
    for sub_id in ds_list["code"]:
        for act_id, act in enumerate(act_labels):
            for trial in trial_codes[act_id]:
                fname = path + 'A_DeviceMotion_data/' + act + '_' + str(trial) + '/sub_' + str(int(sub_id)) + '.csv'
                raw_data = pd.read_csv(fname)
                raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
                vals = np.zeros((len(raw_data), num_data_cols))

                if combine_grav_acc:
                    raw_data["userAcceleration.x"] = raw_data["userAcceleration.x"].add(raw_data["gravity.x"])
                    raw_data["userAcceleration.y"] = raw_data["userAcceleration.y"].add(raw_data["gravity.y"])
                    raw_data["userAcceleration.z"] = raw_data["userAcceleration.z"].add(raw_data["gravity.z"])

                for x_id, axes in enumerate(dt_list):
                    if mode == "mag":
                        vals[:, x_id] = (raw_data[axes] ** 2).sum(axis=1) ** 0.5
                    else:
                        vals[:, x_id * 3:(x_id + 1) * 3] = raw_data[axes].values
                    vals = vals[:, :num_data_cols]
                if labeled:
                    lbls = np.array([[act_id,
                                      sub_id - 1,
                                      ds_list["weight"][sub_id - 1],
                                      ds_list["height"][sub_id - 1],
                                      ds_list["age"][sub_id - 1],
                                      ds_list["gender"][sub_id - 1],
                                      trial
                                      ]] * len(raw_data))
                    vals = np.concatenate((vals, lbls), axis=1)
                dataset = np.append(dataset, vals, axis=0)
    cols = []
    for axes in dt_list:
        if mode == "raw":
            cols += axes
        else:
            cols += [str(axes[0][:-2])]

    if labeled:
        cols += ["act", "id", "weight", "height", "age", "gender", "trial"]

    dataset = pd.DataFrame(data=dataset, columns=cols)
    return dataset

def ts_to_secs(dataset, w, s, standardize=False, **options):
    data = dataset[dataset.columns[:-7]].values
    act_labels = dataset["act"].values
    id_labels = dataset["id"].values
    trial_labels = dataset["trial"].values

    mean = 0
    std = 1
    if standardize:
        ## Standardize each sensor’s data to have a zero mean and unity standard deviation.
        ## As usual, we normalize test dataset by training dataset's parameters
        if options:
            mean = options.get("mean")
            std = options.get("std")
            print("[INFO] -- Test Data has been standardized")
        else:
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            print("[INFO] -- Training Data has been standardized: the mean is = " + str(
                mean) + " ; and the std is = " + str(std))

        data -= mean
        data /= std
    else:
        print("[INFO] -- Without Standardization.....")

    ## We want the Rows of matrices show each Feature and the Columns show time points.
    data = data.T

    m = data.shape[0]  # Data Dimension
    ttp = data.shape[1]  # Total Time Points
    number_of_secs = int(round(((ttp - w) / s)))

    ##  Create a 3D matrix for Storing Sections
    secs_data = np.zeros((number_of_secs, m, w))
    act_secs_labels = np.zeros(number_of_secs)
    id_secs_labels = np.zeros(number_of_secs)

    k = 0
    for i in range(0, ttp - w, s):
        j = i // s
        if j >= number_of_secs:
            break
        if id_labels[i] != id_labels[i + w - 1]:
            continue
        if act_labels[i] != act_labels[i + w - 1]:
            continue
        if trial_labels[i] != trial_labels[i + w - 1]:
            continue

        secs_data[k] = data[:, i:i + w]
        act_secs_labels[k] = act_labels[i].astype(int)
        id_secs_labels[k] = id_labels[i].astype(int)
        k = k + 1

    secs_data = secs_data[0:k]
    act_secs_labels = act_secs_labels[0:k]
    id_secs_labels = id_secs_labels[0:k]
    return secs_data, act_secs_labels, id_secs_labels, mean, std
##________________________________________________________________

class DataSampler(object):
    def __init__(self, path):
        self.path = path

        # self.ACT_LABELS = ["dws", "ups", "wlk", "jog", "std", "sit"]
        self.ACT_LABELS = ["dws", "ups", "wlk", "jog"]

        self.TRIAL_CODES = {
            self.ACT_LABELS[0]: [1, 2, 11],
            self.ACT_LABELS[1]: [3, 4, 12],
            self.ACT_LABELS[2]: [7, 8, 15],
            self.ACT_LABELS[3]: [9, 16],
        }
        self.shape = [2, 128, 1]
        ## Here we set parameter to build labeld time-series from dataset of "(A)DeviceMotion_data"
        ## attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
        self.sdt = ["rotationRate", "userAcceleration"]
        self.mode = "mag"
        self.cga = True  # Add gravity to acceleration or not
        print("[INFO] -- Selected sensor data types: " + str(self.sdt) + " -- Mode: " + str(
            self.mode) + " -- Grav+Acc: " + str(
            self.cga))

        self.act_labels = self.ACT_LABELS[0:4]
        print("[INFO] -- Selected activites: " + str(self.act_labels))
        self.trial_codes = [self.TRIAL_CODES[act] for act in self.act_labels]
        self.dt_list = set_data_types(self.sdt)
        self.dataset = creat_time_series(self.path, self.dt_list, self.act_labels, self.trial_codes, mode=self.mode,
                                            labeled=True, combine_grav_acc=self.cga)
        print("[INFO] -- Shape of time-Series dataset:" + str(self.dataset.shape))

        self.test_trail = [11, 12, 13, 14, 15, 16]
        print("[INFO] -- Test Trials: " + str(self.test_trail))
        self.test_ts = self.dataset.loc[(self.dataset['trial'].isin(self.test_trail))]
        self.train_ts = self.dataset.loc[~(self.dataset['trial'].isin(self.test_trail))]

        print("[INFO] -- Shape of Train Time-Series :" + str(self.train_ts.shape))
        print("[INFO] -- Shape of Test Time-Series :" + str(self.test_ts.shape))

        ## This Variable Defines the Size of Sliding Window
        ## ( e.g. 100 means in each snapshot we just consider 100 consecutive observations of each sensor)
        w = 128  # 50 Equals to 1 second for MotionSense Dataset (it is on 50Hz samplig rate)
        ## Here We Choose Step Size for Building Diffrent Snapshots from Time-Series Data
        ## ( smaller step size will increase the amount of the instances and higher computational cost may be incurred )
        s = 10
        self.train_data, self.act_train, self.id_train, self.train_mean, self.train_std = ts_to_secs(self.train_ts.copy(),
                                                                               w,
                                                                               s,
                                                                               standardize=True)

        s = 10
        self.test_data, self.act_test, self.id_test, self.test_mean, self.test_std = ts_to_secs(self.test_ts.copy(),
                                                                          w,
                                                                          s,
                                                                          standardize=True,
                                                                          mean=self.train_mean,
                                                                          std=self.train_std)

        self.size_train_data = self.train_data.shape[0]
        self.train_data = np.reshape(self.train_data, [self.size_train_data, 256])
        #
        self.size_test_data = self.test_data.shape[0]
        self.test_data = np.reshape(self.test_data, [self.size_test_data, 256])

        self.act_train_labels = to_categorical(self.act_train)
        self.act_test_labels = to_categorical(self.act_test)
        self.id_train_labels = to_categorical(self.id_train)
        self.id_test_labels = to_categorical(self.id_test)

        data_subject_info = pd.read_csv(self.path + "data_subjects_info.csv")
        id_gen_info = data_subject_info[["code", "gender"]].values
        gen_id_dic = {item[0]: item[1] for item in id_gen_info}

        tmp = self.id_train.copy()
        gen_train = np.array([gen_id_dic[item + 1] for item in tmp])
        self.gen_train_labels = (gen_train).copy()
        self.gen_train_labels = to_categorical(self.gen_train_labels, num_classes=2)

        tmp = self.id_test.copy()
        gen_test = np.array([gen_id_dic[item + 1] for item in tmp])
        self.gen_test_labels = (gen_test).copy()
        self.gen_test_labels = to_categorical(self.gen_test_labels, num_classes=2)

    def next_batch(self, num, data, labels):
        '''
        Return a total of `num` random samples and labels.
        '''
        idx = np.arange(0, len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[i] for i in idx]
        labels_shuffle = [labels[i] for i in idx]
        return np.asarray(data_shuffle), np.asarray(labels_shuffle)

    def train(self, batch_size, label=False):
        if label:
            return self.next_batch(batch_size, self.train_data, self.train_labels)
        else:
            return self.next_batch(batch_size, self.train_data, self.train_labels)[0]
    def get_act_test(self):
        return self.act_test
    def get_act_train(self):
        return self.act_train
    def w_all_test_gender(self):
        return self.w_gen_test_labels
    def all_train(self):
        return self.train_data
    def all_train_labels(self):
        return self.act_train_labels
    def all_gender_train_labels(self):
        return self.gen_train_labels
    def all_gender_test_labels(self):
        return self.gen_test_labels
    def all_test(self):
        return self.test_data
    def all_test_labels(self):
        return self.act_test_labels

def load_motionsense():
    path = "../../../Dataset&Models/MotionSense Dataset/"
    DS = DataSampler(path)
    user_groups = {}  # Generate User groups: dictionary
    user_groups_test = {}

    x_train = DS.all_train()
    act_train_labels = DS.all_train_labels()
    gen_train_labels = DS.all_gender_train_labels()
    user_id_train = DS.id_train
    
    train_id = np.arange(len(x_train))

    for user_id in range(24):
        user_filter = user_id_train[:]==user_id
        user_groups[user_id] = train_id[user_filter]

    x_test = DS.all_test()
    act_test_labels = DS.all_test_labels()
    gen_test_labels = DS.all_gender_test_labels()
    user_id_test = DS.id_test # user id for each data sample
    test_id = np.arange(start=len(x_train) ,stop=(len(x_test) + len(x_train))) # id for each data sample
    
    for user_id in range(24):
        user_filter = user_id_test[:]==user_id
        user_groups_test[user_id] = test_id[user_filter]

    return x_train, x_test, act_train_labels, act_test_labels, gen_train_labels, gen_test_labels, user_groups, user_groups_test, train_id.astype('int'), test_id.astype('int')


# MobiAct Preprocessing
def load_mobiact(args):
    path="../../../Dataset&Models/MobiAct Dataset/"
    data_subjects = pd.read_csv(path + "data_subjects.csv")
    data = np.load(path + "Data/total_data.npy", allow_pickle=True)
    activity = np.load(path + "Data/activity_labels.npy", allow_pickle=True)
    gender = np.load(path + "Data/gender_labels.npy", allow_pickle=True)
    id = np.load(path + "Data/id_labels.npy", allow_pickle=True)
    weight = np.load(path + "Data/weights_data.npy", allow_pickle=True)

    
    # shuffle data
    array = np.arange(data.shape[0])
    np.random.seed(args.seed)
    np.random.shuffle(array)

    data = data[array]
    activity = activity[array]
    gender = gender[array]
    weight = weight[array]
    
    for i in range(len(weight)):
        if weight[i] <= 70:
            weight[i] = 0
        elif weight[i] <= 90:
            weight[i] = 1
        else:
            weight[i] = 2    
    
    id = id[array]
    data_train = np.array([]).reshape(0, data.shape[1], data.shape[2]) # data_train.shape=(365452, 6, 128)
    data_test = np.array([]).reshape(0, data.shape[1], data.shape[2])
    activity_train = np.array([])
    activity_test = np.array([])
    gender_train = np.array([])
    gender_test = np.array([])
    id_train = np.array([])
    id_test = np.array([])
    weight_train = np.array([])
    weight_test = np.array([])
    
    # for each user id, select their shuffled data, age, activity, gender samples,
    # first split tran-test set, then concatenate by the order of subject id    
    # id[:] == i -> [False, False, False, ...., False]   
    user_groups={}     # Generate User groups: dictionary
    user_groups_test = {}
    
    counter=0
    for i in data_subjects["id"]:
        id_filter = id[:] == i
        sub_id = array[id_filter]
        data_sub_id = data[id_filter]
        activity_sub_id = activity[id_filter]
        gender_sub_id = gender[id_filter]
        weight_sub_id = weight[id_filter]
        x_train, x_test, act_train, act_test, gen_train, gen_test, wgh_train, wgh_test, sub_id_train, sub_id_test = train_test_split(data_sub_id, activity_sub_id, gender_sub_id, weight_sub_id, sub_id, test_size = 0.2, random_state = args.seed)
        data_train = np.concatenate((data_train, x_train), axis=0)
        data_test = np.concatenate((data_test, x_test), axis=0)
        activity_train = np.concatenate((activity_train, act_train), axis=0)
        activity_test = np.concatenate((activity_test, act_test), axis=0)
        gender_train = np.concatenate((gender_train, gen_train), axis=0)
        gender_test = np.concatenate((gender_test, gen_test), axis=0)
        
        weight_train = np.concatenate((weight_train, wgh_train), axis=0)
        weight_test = np.concatenate((weight_test, wgh_test), axis=0)
        
        id_train = np.concatenate((id_train, sub_id_train), axis=0)
        id_test = np.concatenate((id_test, sub_id_test), axis=0)        
        user_groups[counter]=sub_id_train # Key: subject id, -1 because csv starts at 1, Values: index in the dataset
        user_groups_test[counter] = sub_id_test
        counter+=1
        
    # Count the number of unique classes and generate one-hot labels
    nb_classes = len(np.unique(activity_train[:]))
    activity_train_label = to_categorical(activity_train[:], nb_classes) # shape=(365452, 4)
    nb_classes = len(np.unique(activity_test[:]))
    activity_test_label = to_categorical(activity_test[:], nb_classes)
    nb_classes = len(np.unique(gender_train[:]))
    gender_train_label = to_categorical(gender_train, nb_classes)   # shape=(365452,), 1d
    nb_classes = len(np.unique(gender_test[:]))
    gender_test_label = to_categorical(gender_test, nb_classes)   
    nb_classes = len(np.unique(weight_train[:]))
    weight_train_label = to_categorical(weight_train, nb_classes)   # shape=(365452,), 1d
    nb_classes = len(np.unique(weight_test[:]))
    weight_test_label = to_categorical(weight_test, nb_classes)     
    
    x_train = data_train.reshape((data_train.shape[0], data_train.shape[1], data_train.shape[2], 1)) # shape=(365452, 6, 128, 1)
    x_test = data_test.reshape((data_test.shape[0], data_test.shape[1], data_test.shape[2], 1))

    return x_train, x_test, activity_train_label, activity_test_label, gender_train_label, gender_test_label, weight_train_label, weight_test_label, user_groups, user_groups_test, id_train.astype('int'), id_test.astype('int')
