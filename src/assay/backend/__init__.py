# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Vendor-neutral backend protocols; concrete adapters are consumer-owned."""

from .features import (
    BACKEND_CONFORMANCE_VERSION,
    BACKEND_FEATURE_SCHEMA_VERSION,
    BackendFeatureSet,
    CONSERVATIVE_BACKEND_FEATURES,
    FeatureAwareBackend,
)
from .protocol import Backend, ChatToolsBackend

__all__ = [
    "BACKEND_CONFORMANCE_VERSION",
    "BACKEND_FEATURE_SCHEMA_VERSION",
    "Backend",
    "BackendFeatureSet",
    "CONSERVATIVE_BACKEND_FEATURES",
    "ChatToolsBackend",
    "FeatureAwareBackend",
]
