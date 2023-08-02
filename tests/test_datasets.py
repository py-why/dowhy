import pandas as pd
import pytest
from flaky import flaky
from pytest import mark
from sklearn import linear_model
import numpy as np
import dowhy
from dowhy.datasets import sales_dataset

class TestDataset(object):

    def test_sales_dataset(self, max_shop_events:int=10):
        sales_df = sales_dataset("2022-01-01","2022-06-01",max_shop_events=max_shop_events)

        assert sales_df.shape[0]>0

        # check basic stuff

        for col in ['Page Visit','Unit Sold']:
            assert all(np.where(sales_df[col] > 0, True, False))

        for col in ['Price','Ad Spend','Operation Cost','Revenue']:
            assert all(np.where(sales_df[col] > 0.0, True, False))
