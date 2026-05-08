from .DIN_trainer import DINTrain
from .DIN_Model import DeepInterestNetwork
from .generate_training_batches import Train_instance

try:
    from .Trm4Rec_trainer import Trm4Rec
except ModuleNotFoundError as exc:
    if exc.name != 'transformers':
        raise
    Trm4Rec = None

try:
    from .JTM_variant import JTM_Variant
except ModuleNotFoundError as exc:
    if exc.name not in {'lib.JTM_variant', 'JTM_variant'}:
        raise
    JTM_Variant = None
