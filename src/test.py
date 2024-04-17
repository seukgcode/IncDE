from .utils import *
from .model.model_process import *

class Tester():
    def __init__(self, args, kg, model) -> None:
        self.args = args
        self.kg = kg
        self.model = model
        self.test_processor = DevBatchProcessor(args, kg)

    def test(self):
        self.args.valid = False
        return self.test_processor.process_epoch(self.model)