def get_model(model_name, args):
    name = model_name.lower()
    if name in {'mmle', 'm2le'}:
        from methods.mmle import Learner
        return Learner(args)
    raise ValueError(f"Unknown model '{model_name}'. Expected 'mmle' or 'm2le'.")
