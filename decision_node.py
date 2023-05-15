import pandas as pd
import numpy as np

class TreeNode():
    # I take the value, children and the feature of the node which the node is split from
    def __init__(self, l_child=None, r_child=None, feature=None, threshold=None, val=None):
        self.l_child = l_child
        self.r_child = r_child
        self.feature = feature
        self.val = val
        self.threshold = threshold

    def is_leaf(self):
        if self.l_child is None and self.r_child is None:
            return True
        return False
