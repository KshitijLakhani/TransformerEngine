# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Pytest configuration for multi-process context parallel tests"""
import pytest


def pytest_addoption(parser):
    """Add multi-process options for context parallel tests"""
    parser.addoption("--num-process", action="store", default=0)
    parser.addoption("--process-id", action="store", default=0)


@pytest.fixture(autouse=True)
def multiprocessing_parses(request):
    """Fixture for querying num-process and process-id"""
    if request.cls:
        request.cls.num_process = int(request.config.getoption("--num-process"))
        request.cls.process_id = int(request.config.getoption("--process-id"))
