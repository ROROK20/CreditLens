"""Suite-wide settings.

POST /train is off unless the operator sets CREDITLENS_ENABLE_TRAIN=1
(creditlens/api/app.py). The suite exercises it, so it opts in here;
tests/test_train_endpoint.py removes the opt-in to prove the default is off.
"""

import os

os.environ["CREDITLENS_ENABLE_TRAIN"] = "1"
