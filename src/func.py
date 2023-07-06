import os
from pathlib import Path
from typing import Optional
from symai import Symbol, Expression
from symai.components import TokenTracker
from symai.extended.conversation import Conversation


class SymAsk(Expression):
    def __init__(self):
        super().__init__()
        # get current file location as absolute path
        self.temp_path = Path(__file__).parent.absolute() / '../.tmp/'
        os.makedirs(self.temp_path, exist_ok=True)
        self.temp_file = self.temp_path / 'symask.pkl'
        if os.path.exists(self.temp_file):
            obj = Symbol.load(self.temp_file)
            self.conv = Conversation(str(obj), auto_print=False)
            self.conv._memory = str(obj)
        else:
            self.conv = Conversation(auto_print=False)

    def forward(self, query: Optional[str] = None, fn: Optional[str] = None, *args, **kwargs):
        query = self._to_symbol(query)
        if 'init' in kwargs or 'drop' in kwargs or 'reset' in kwargs:
            self.conv.drop()
            self.conv.save(self.temp_file, replace=True)
            if 'drop' in kwargs:
                del kwargs['drop']
            if 'reset' in kwargs:
                del kwargs['reset']
            if 'init' in kwargs:
                self.conv.store_system_message(kwargs['init'])
                del kwargs['init']

        if 'forget' in kwargs:
            self.conv.drop()
            self.conv.forget(kwargs['forget'])
            del kwargs['forget']

        if 'file' in kwargs:
            self.conv.store_file(kwargs['file'])
            del kwargs['file']

        if query is None:
            return ''
        with TokenTracker() as tracker:
            res = self.conv(query, *args, **kwargs)
        print(tracker)
        self.conv.save(self.temp_file, replace=True)
        return f"{res}\n{tracker}" 
