#!/usr/bin/env python3

import numpy as np
import pytest

import torch

from findview_baselines.common.tensor_dict import TensorDict


def test_tensor_dict_constructor():
    dict_tree = dict(
        a=torch.randn(2, 2), b=dict(c=dict(d=np.random.randn(3, 3)))
    )
    tensor_dict = TensorDict.from_tree(dict_tree)

    assert torch.is_tensor(tensor_dict["a"])
    assert isinstance(tensor_dict["b"], TensorDict)
    assert isinstance(tensor_dict["b"]["c"], TensorDict)
    assert torch.is_tensor(tensor_dict["b"]["c"]["d"])


def test_tensor_dict_to_tree():
    dict_tree = dict(a=torch.randn(2, 2), b=dict(c=dict(d=torch.randn(3, 3))))

    assert dict_tree == TensorDict.from_tree(dict_tree).to_tree()


def test_tensor_dict_str_index():
    dict_tree = dict(a=torch.randn(2, 2), b=dict(c=dict(d=torch.randn(3, 3))))
    tensor_dict = TensorDict.from_tree(dict_tree)

    x = torch.randn(5, 5)
    tensor_dict["a"] = x
    assert (tensor_dict["a"] == x).all()

    with pytest.raises(KeyError):
        _ = tensor_dict["c"]


def test_tensor_dict_index():
    dict_tree = dict(a=torch.randn(2, 2), b=dict(c=dict(d=torch.randn(3, 3))))
    tensor_dict = TensorDict.from_tree(dict_tree)

    with pytest.raises(KeyError):
        tensor_dict["b"][0] = dict(q=torch.randn(3))

    tmp = dict(c=dict(d=torch.randn(3)))
    tensor_dict["b"][0] = tmp
    assert torch.allclose(tensor_dict["b"]["c"]["d"][0], tmp["c"]["d"])
    assert not torch.allclose(tensor_dict["b"]["c"]["d"][1], tmp["c"]["d"])

    tensor_dict["b"]["c"]["x"] = torch.randn(5, 5)
    with pytest.raises(KeyError):
        tensor_dict["b"][1] = tmp

    tensor_dict["b"].set(1, tmp, strict=False)
    assert torch.allclose(tensor_dict["b"]["c"]["d"][1], tmp["c"]["d"])

    tmp = dict(c=dict(d=torch.randn(1, 3)))
    del tensor_dict["b"]["c"]["x"]
    tensor_dict["b"][2:3] = tmp
    assert torch.allclose(tensor_dict["b"]["c"]["d"][2:3], tmp["c"]["d"])


def test_tensor_dict_map():
    dict_tree = dict(a=dict(b=[0]))
    tensor_dict = TensorDict.from_tree(dict_tree)

    res = tensor_dict.map(lambda x: x + 1)
    assert (res["a"]["b"] == 1).all()

    tensor_dict.map_in_place(lambda x: x + 1)

    assert res == tensor_dict
