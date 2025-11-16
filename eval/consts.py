import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum


class PredictionKeys(StrEnum):
    MODEL_PREDICTION = "prediction"
    GT_LETTER = "gt_letter"
    GT_TEXT = "gt_text"
    MANEUVER_ID = "man_id"


class ResultKeys(StrEnum):
    EGO = "ego"
    EGO_AGENT = "ego_agent"
    AGENT = "agent"
    AGENT_AGENT = "agent_agent"
