def parse_state(state):
    if isinstance(state, (str, int, float)):
        return [state]
    if isinstance(state, list):
        return state
    if isinstance(state, dict):
        return [xi for xi in state.keys()]
    if not state:
        return []
    raise Exception("Input format for {} not recognized: {}".format(state, type(state)))
