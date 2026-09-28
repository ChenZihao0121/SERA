# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod

from verl import DataProto

__all__ = ["BaseRollout", "get_rollout_n"]


def get_rollout_n(prompts: DataProto, default_n: int) -> int:
    """Return the number of responses requested for this rollout call.

    ``rollout_n_override`` is intentionally call-scoped through ``meta_info``.
    It lets two-stage trainers use the backend's native multi-sample path for
    a fixed-size pilot and switch to one response per explicitly repeated
    continuation row without mutating the global rollout configuration.
    """
    rollout_n = prompts.meta_info.get("rollout_n_override", default_n)
    if isinstance(rollout_n, bool) or not isinstance(rollout_n, int) or rollout_n < 1:
        raise ValueError(f"rollout_n_override must be a positive integer, got {rollout_n!r}")
    return rollout_n


class BaseRollout(ABC):
    """Base class for rollout."""

    @abstractmethod
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences"""
        pass
