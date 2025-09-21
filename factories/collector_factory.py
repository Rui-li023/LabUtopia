from typing import Dict, Type
from data_collectors.data_collector import DataCollector
from data_collectors.mock_collector import MockCollector

_collector_registry: Dict[str, Type[DataCollector]] = {}

def register_collector(name: str, collector_class: Type[DataCollector]):
    
    _collector_registry[name] = collector_class

def create_collector(collector_type: str, *args, **kwargs) -> DataCollector:
    
    if collector_type not in _collector_registry:
        raise ValueError(f": {collector_type}")
    return _collector_registry[collector_type](*args, **kwargs)


register_collector("default", DataCollector)
register_collector("mock", MockCollector)
