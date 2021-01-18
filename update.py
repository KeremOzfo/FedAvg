
def pull_model(model_user, model_server):
    for param_user, param_server in zip(model_user.parameters(), model_server.parameters()):
        param_user.data = param_server.data[:] + 0
    return None
