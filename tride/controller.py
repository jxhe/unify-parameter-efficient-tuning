import numpy as np


def override(hidden_states, override):
    """chanage the hidden states according to
    the arguments specified by override dict

    Args:

    hidden_states (tensor): (B, T, H)
    override (dict)
    """

    alpha = override['alpha']
    sklearn_model = override['model']
    index = sklearn_model.coef_[0].nonzero()[0]
    w = sklearn_model.coef_[0][index]
    b = sklearn_model.intercept_[0]

    hidden_states_cpu = hidden_states.cpu().numpy()

    x = hidden_states_cpu[:,:,index]

    w_expand = np.expand_dims(w, axis=(0,1))

    project_x = x - ((np.dot(x, w) + b) / np.sqrt(np.dot(w, w))) * w_expand

    final_x = project_x + alpha * w_expand

    hidden_states_cpu[:,:,index] = final_x

    return hidden_states.new_tensor(hidden_states_cpu)
