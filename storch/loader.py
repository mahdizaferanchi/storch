from .helpers import DummyLogger, move_to, configurable
from .exec_info import device
import sys

inf = sys.maxsize

@configurable
class EnumData:
    def __init__(
        self,
        dataloader,
        steps=inf,
        ignore_depletion=False,
        reset_on_start=True,
        logger=DummyLogger(),
    ):
        self.source = dataloader
        self.steps = steps
        self.reset_on_start = reset_on_start
        self.ignore_depletion = ignore_depletion
        # if not self.reset_on_start:
        self.iterator = iter(self.source)
        self.step = 0
        self.logger = logger
        try:
            logger.set_enumerator_len(len(self))
        except ValueError:
            pass

    def force_next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.source)
            return next(self.iterator)

    def __iter__(self):
        if self.reset_on_start:
            self.iterator = iter(self.source)
        self.step = 0
        return self

    def __len__(self):
        if self.ignore_depletion:
            value = self.steps
        else:
            if not self.reset_on_start:
                value = inf
            else:
                try:
                    source_len = len(self.source)
                except:
                    source_len = inf
                source_len = source_len or inf
                # value = inf if source_len == inf else min(
                #     source_len, self.steps)
                value = min(source_len, self.steps)
        if value == inf:
            raise ValueError()
        return value

    def get_next(self):
        self.step += 1
        if self.step > self.steps:
            raise StopIteration
        if self.ignore_depletion or self.step == 1:
            return self.force_next()
        return next(self.iterator)

    def __next__(self):
        value = self.get_next()
        self.logger(self)
        return move_to(value, device)
