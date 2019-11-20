import numpy as np

ACTION_LONG = 1
ACTION_SHORT = 2

ACTION_SET = [ACTION_LONG, ACTION_SHORT]


def extract_features_from_row_tuple(row_tuple, feature_names):
    features = [
        f"{feature_name}:{getattr(row_tuple, feature_name)}"
        for feature_name in feature_names
    ]
    return " ".join(features)


def sample_an_action(pmf):
    pmf = [np.round(n, 4) for n in pmf]
    action = np.random.choice(ACTION_SET, p=pmf)
    prob = pmf[action - 1]

    return action, prob


def compute_cost(action, label):
    if action == ACTION_LONG and label == 1:
        return -1
    elif action == ACTION_SHORT and label == -1:
        return -1
    else:
        return 0


def build_example(action, cost, prob, features):
    return f"{action}:{cost}:{prob} | {features}"
