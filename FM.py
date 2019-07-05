import pandas as pd
import xlearn as xl
import warnings

train = pd.read_csv('diet_data.csv')
warnings.filterwarnings('ignore')

del train['Unnamed: 0']

dict_ls = {'Amc': 2, 'Med': 1, 'Veg': 0}

dict_lss = {'Over': 1, 'Lean': 0}

train['diet'].replace(dict_ls, inplace=True)

train['obs'].replace(dict_lss, inplace=True)

from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(train, test_size = 0.3, random_state = 5)

def convert_to_ffm(df, type, numerics, categories, features):
    currentcode = len(numerics)
    catdict = {}
    catcodes = {}

    # Flagging categorical and numerical fields
    for x in numerics:
        catdict[x] = 0
    for x in categories:
        catdict[x] = 1

    nrows = df.shape[0]
    ncolumns = len(features)
    with open(str(type) + "_ffm.txt", "w") as text_file:

        # Looping over rows to convert each row to libffm format
        for n, r in enumerate(range(nrows)):
            datastring = ""
            datarow = df.iloc[r].to_dict()
            datastring += str(int(datarow['obs']))  # Set Target Variable here

            # For numerical fields, we are creating a dummy field here
            for i, x in enumerate(catdict.keys()):
                if (catdict[x] == 0):
                    datastring = datastring + " " + str(i) + ":" + str(i) + ":" + str(datarow[x])
                else:

                    # For a new field appearing in a training example
                    if (x not in catcodes):
                        catcodes[x] = {}
                        currentcode += 1
                        catcodes[x][datarow[x]] = currentcode  # encoding the feature

                    # For already encoded fields
                    elif (datarow[x] not in catcodes[x]):
                        currentcode += 1
                        catcodes[x][datarow[x]] = currentcode  # encoding the feature

                    code = catcodes[x][datarow[x]]
                    datastring = datastring + " " + str(i) + ":" + str(int(code)) + ":1"

            datastring += '\n'
            text_file.write(datastring)

convert_to_ffm(X_train,'Train',{'PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14','PC15','PC16','PC17','PC18','PC19','PC20'},{'Sex','diet'},{'PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14','PC15','PC16','PC17','PC18','PC19','PC20','Sex','diet'})

convert_to_ffm(X_test,'Test',{'PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14','PC15','PC16','PC17','PC18','PC19','PC20'},{'Sex','diet'},{'PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14','PC15','PC16','PC17','PC18','PC19','PC20','Sex','diet'})

ffm_model = xl.create_ffm()

ffm_model.setTrain("Train_ffm.txt")



param = {'task':'binary',
         'lr':0.2,
         'lambda':0.002,
         'metric':'acc'}

# Start to train
# The trained model will be stored in model.out
ffm_model.fit(param, './model.out')

# Prediction task
ffm_model.setTest("Test_ffm.txt") # Test data
ffm_model.setSigmoid() # Convert output to 0-1

# Start to predict
# The output result will be stored in output.txt
ffm_model.predict("./model.out", "./output.txt")

ffm_model.cv(param)


