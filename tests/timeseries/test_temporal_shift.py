import unittest
import pandas as pd
from typing import List, Optional

from dowhy.timeseries.temporal_shift import shift_columns_by_lag

class TestShiftColumnsByLag(unittest.TestCase):

    def test_shift_columns_by_lag_basic_shift(self):
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1]
        })
        columns = ['A', 'B']
        lag = [2, 2]
        result = shift_columns_by_lag(df, columns, lag, filter=False)
        
        expected = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1],
            'A_lag1': [0, 1, 2, 3, 4],
            'A_lag2': [0, 0, 1, 2, 3],
            'B_lag1': [0, 5, 4, 3, 2],
            'B_lag2': [0, 0, 5, 4, 3]
        })
        
        pd.testing.assert_frame_equal(result, expected)

    def test_shift_columns_by_lag_different_lags(self):
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1]
        })
        columns = ['A', 'B']
        lag = [1, 3]
        result = shift_columns_by_lag(df, columns, lag, filter=False)
        
        expected = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1],
            'A_lag1': [0, 1, 2, 3, 4],
            'B_lag1': [0, 5, 4, 3, 2],
            'B_lag2': [0, 0, 5, 4, 3],
            'B_lag3': [0, 0, 0, 5, 4]
        })
        
        pd.testing.assert_frame_equal(result, expected)

    def test_shift_columns_by_lag_with_filter(self):
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1],
            'C': [1, 1, 1, 1, 1]
        })
        columns = ['A', 'B']
        lag = [1, 2]
        result = shift_columns_by_lag(df, columns, lag, filter=True, child_node='C')
        
        expected = pd.DataFrame({
            'C': [1, 1, 1, 1, 1],
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1],
            'A_lag1': [0, 1, 2, 3, 4],
            'B_lag1': [0, 5, 4, 3, 2],
            'B_lag2': [0, 0, 5, 4, 3]
        })
        
        pd.testing.assert_frame_equal(result, expected)

    def test_shift_columns_by_lag_empty_dataframe(self):
        df = pd.DataFrame(columns=['A', 'B'])
        columns = ['A', 'B']
        lag = [2, 2]
        result = shift_columns_by_lag(df, columns, lag, filter=False)
        
        expected = pd.DataFrame(columns=['A', 'B', 'A_lag1', 'A_lag2', 'B_lag1', 'B_lag2'])
        
        pd.testing.assert_frame_equal(result, expected)

    def test_shift_columns_by_lag_unequal_columns_and_lags(self):
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1]
        })
        columns = ['A', 'B']
        lag = [1]
        
        with self.assertRaises(ValueError) as context:
            shift_columns_by_lag(df, columns, lag, filter=False)
        
        self.assertTrue("The size of 'columns' and 'lag' lists must be the same." in str(context.exception))

if __name__ == '__main__':
    unittest.main()
