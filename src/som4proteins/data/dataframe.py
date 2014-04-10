'''dataframe module

'''
import numpy as np

class DataFrame:
    '''Contains the data to be processed in a matrix form.
    
    :param list_of_lists_of_numbers data: Matrix containing the data.
    :param list_of_strings row_labels: Names of the columns.
    :param list_of_strings col_labels: Names of the rows.
    
    :ivar array_of_strings row_labels: Names of the rows.
    :ivar array_of_strings col_labels: Names of the columns.
    :ivar int n_cols: Number of columns.
    :ivar int n_rows: Number of rows.
    :ivar 2D_array_of_numbers data: Data.
    
    '''
    
    def __init__(self, data=[], row_labels=[], col_labels=[]):
        self._data = np.array(data)
        self._row_labels = np.array(row_labels)
        self._col_labels = np.array(col_labels)
        
    @property
    def row_labels(self):
        return self._row_labels
    
    @row_labels.setter
    def row_labels(self, orow_labels):
        self._row_labels = np.array(orow_labels)
    
    @property
    def col_labels(self):
        return self._col_labels
    
    @col_labels.setter
    def col_labels(self, ocol_labels):
        self._col_labels = np.array(ocol_labels)
    
    @property
    def n_cols(self):
        return self._data.shape[1]
    
    @property
    def n_rows(self):
        return self._data.shape[0]
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, odata):
        self._data = np.array(odata)
    
    def __str__(self):
        output_str = "Column labels:" + str(self.col_labels)
        output_str += "\nRow labels:" + str(self.row_labels)
        return output_str + "\nData:\n" + str(self.data) + "\n"
    
    def join(self, odataframe):
        if len(self.data) == 0 and len(self.row_labels) == 0 and len(self.col_labels) == 0:
            self._data = odataframe.data
            self._row_labels = odataframe.row_labels
            self._col_labels = odataframe.col_labels
        else:
            self.data = np.vstack((self.data, odataframe.data))
            self.row_labels = np.concatenate((self.row_labels, odataframe.row_labels))
