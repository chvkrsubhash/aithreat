import numpy as np
import pandas as pd
import traceback
from utils.ml_logic import run_inference, get_shap_explanation

# The model expects 78 features according to standard CIC-IDS-2017 dataset
df = pd.DataFrame(np.random.rand(1, 78), columns=[str(i) for i in range(78)])

try:
    print("Running inference...")
    label, conf, probs = run_inference("Best Model", df)
    print("Inference successful. Label:", label)
    
    print("Running SHAP...")
    plot = get_shap_explanation("Best Model", df)
    if plot:
        print("SHAP successful.")
except Exception as e:
    print("ERROR CAUGHT:")
    traceback.print_exc()
