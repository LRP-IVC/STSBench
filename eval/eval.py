import json

import numpy as np
from prettytable import PrettyTable
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from annotator.data.maneuvers import Maneuver, ManeuverType
from annotator.data.models import Frame

from .consts import PredictionKeys, ResultKeys


def find_answer_by_id(man_id, answers):
    for a in answers:
        if man_id == a[PredictionKeys.MANEUVER_ID]:
            return a
    assert False, f"Maneuver with id {man_id} from DB not found in generated answers"


def is_correct(answer, gt):
    return answer[0].lower() == gt[0].lower()


def evaluate(args):
    with open(args.output_path, "r") as f:
        answers = json.load(f)

    engine = create_engine(f"sqlite:///{str(args.db_path.name)}", echo=False)
    session = Session(engine)

    results = {
        ResultKeys.EGO: {},
        ResultKeys.EGO_AGENT: {},
        ResultKeys.AGENT: {},
        ResultKeys.AGENT_AGENT: {},
    }
    stmt = select(Maneuver).where(
        Maneuver.manually_labeled,
        Maneuver.in_use,
        Maneuver.pos_maneuvers != None,
        Maneuver.pos_maneuvers.any(),
    )
    mans = session.scalars(stmt).all()
    for man in mans:
        man_type = man.pos_maneuvers[0].type

        a = find_answer_by_id(man.id, answers)

        man_group = None
        if man.is_ego and not man.is_other_agent:
            man_group = ResultKeys.EGO
        if man.is_ego and man.is_other_agent:
            man_group = ResultKeys.EGO_AGENT
        if man.is_agent and not man.is_other_agent:
            man_group = ResultKeys.AGENT
        if man.is_agent and man.is_other_agent:
            man_group = ResultKeys.AGENT_AGENT

        assert man_group is not None, "Unkown maneuver group"

        if man_type not in results[man_group]:
            results[man_group][man_type] = []

        results[man_group][man_type].append(
            is_correct(a[PredictionKeys.MODEL_PREDICTION], a[PredictionKeys.GT_LETTER])
        )

    res_category_man_mean = {
        ResultKeys.EGO: {},
        ResultKeys.EGO_AGENT: {},
        ResultKeys.AGENT: {},
        ResultKeys.AGENT_AGENT: {},
    }
    for man_cat in results.keys():
        for man in results[man_cat]:
            res_category_man_mean[man_cat][man] = np.mean(results[man_cat][man])

    res_category_mean = {
        ResultKeys.EGO: 0,
        ResultKeys.EGO_AGENT: 0,
        ResultKeys.AGENT: 0,
        ResultKeys.AGENT_AGENT: 0,
    }
    for man_cat in res_category_mean.keys():
        res_category_mean[man_cat] = np.mean(
            list(res_category_man_mean[man_cat].values())
        )

    table = PrettyTable()
    table.field_names = [
        col_title.replace("_", " ").title() for col_title in res_category_mean.keys()
    ] + ["Average"]
    for fn in table.field_names:
        table.custom_format[fn] = lambda _, v: f"{v * 100:.2f}"
    table.add_row(
        list(res_category_mean.values()) + [np.mean(list(res_category_mean.values()))]
    )
    print(table)
