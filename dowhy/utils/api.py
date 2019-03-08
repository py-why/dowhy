def parse_state(state):
    if type(state) == str:
        return [state]
    if type(state) == list:
        return state
    if type(state) == dict:
        return [xi for xi in state.keys()]
    if not state:
        return []
    raise Exception('Input format for {} not recognized: {}'.format(state, type(state)))
