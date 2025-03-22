#!/usr/bin/env python3

import great_expectations as gx

print(gx.__version__)

import unittest
import pandas as pd


class TestDatasetQuality(unittest.TestCase):

    def test_no_missing_values(self):
        raise Exception("failed")

if __name__ == "__main__":
    unittest.main()
