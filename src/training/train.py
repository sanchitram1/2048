def main():
    print("Training algorithm goes here")


"""
- step() currently returns (obs, reward, done, truncated, info) (Gymnasium-style). Use that contract consistently in train.py.
- observations are log2 board exponents; make sure your model input expects that.
- invalid actions raise ValueError; either avoid them via legal_actions() or handle exceptions.
- reward config is heuristic; tune later once you see learning curves.
"""
