#Import all required packages and modules 
import matplotlib.pyplot as plt
import pandas as pd
###### functions to use ######


#__ getting data from analysis
def get_data():
    
    train_data = pd.read_csv('Data/train.csv')
    test_data = pd.read_csv('Data/test.csv')
    result = (train_data, test_data)
    return  result


# pick a large window size of 50 cycles
sequence_length = 50

# function to reshape features into (samples, time steps, features) 
def reshapeFeatures(id_df, seq_length, seq_cols):
    """
    Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length.
    An alternative would be to pad sequences so that
    we can use shorter ones.
    
    :param id_df: the data set to modify
    :param seq_length: the length of the window
    :param seq_cols: the columns concerned by the step
    :return: a generator of the sequences
    """
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]

def timeEvolution(turb, train):
    plt.figure(1)
    i = 1
    for col in train.columns[2:]:
        plt.subplot(4, 6, i)
        plt.plot(train[train["id"] == turb].cycle, train[train["id"] == turb][col])
        # plt.plot(test[test["id"] == turb].cycle, test[test["id"] == turb][col], color='red')
        plt.title(col)
        plt.subplots_adjust(top=2, bottom=0.1, left=0.1, right=2, hspace=0.6, wspace=0.8)
        i += 1
    plt.show()
    
# function to generate label
def reshapeLabel(id_df, seq_length=sequence_length, label=['RUL']):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length."""
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length: num_elements, :]
