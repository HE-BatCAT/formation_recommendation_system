"""Neural ageing surrogates (AMS-CeNN, GRU, MLP, linear) for the formation recommendation system.

They take the same formation features as the XGBoost surrogate and return the same four targets, so
``SurrogatePredictor`` can serve either backend. PyTorch is imported only when a neural model is used.
"""
