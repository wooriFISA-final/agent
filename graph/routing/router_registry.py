"""
Router Registry Module

Provides a centralized registry for router classes, similar to the AgentRegistry.
This allows for easy discovery and instantiation of routers based on a name,
making the graph construction process more robust and less prone to import errors.
"""
from typing import Dict, Type, Optional, List
import logging
import importlib
import inspect
import pkgutil

from .router_base import RouterBase

logger = logging.getLogger(__name__)

class RouterRegistry:
    """A singleton registry for router classes."""
    
    _instance = None
    _routers: Dict[str, Type[RouterBase]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, name: Optional[str] = None):
        """
        A class decorator to register a router.

        Example:
            @RouterRegistry.register("my_custom_router")
            class MyCustomRouter(RouterBase):
                ...
        """
        def decorator(router_class: Type[RouterBase]):
            router_name = name or router_class.__name__
            if router_name in cls._routers:
                logger.warning(f"⚠️ Router '{router_name}' is already registered. Overwriting.")
            cls._routers[router_name] = router_class
            logger.info(f"✅ Router registered: {router_name}")
            return router_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[RouterBase]:
        """
        Retrieves a router class by its registered name.
        """
        if name not in cls._routers:
            logger.error(f"Router '{name}' not found. Available routers: {cls.list_routers()}")
            raise KeyError(f"Router '{name}' not found in registry.")
        return cls._routers[name]

    @classmethod
    def list_routers(cls) -> List[str]:
        """Returns a list of all registered router names."""
        return list(cls._routers.keys())

    @classmethod
    def auto_discover(cls, package_name: str = "graph.routing"):
        """
        Automatically discovers and registers routers from a given package.
        It looks for classes that are subclasses of RouterBase.
        """
        try:
            package = importlib.import_module(package_name)
            logger.info(f"🔍 Discovering routers in package: {package_name}")
        except ModuleNotFoundError:
            logger.error(f"❌ Could not find package '{package_name}' for router auto-discovery.")
            return

        for _, module_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            try:
                module = importlib.import_module(module_name)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, RouterBase) and obj is not RouterBase and name not in cls._routers:
                        # Use the class name as the registration name
                        cls._routers[name] = obj
                        logger.info(f"   -> Auto-registered router: {name} from {module_name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load or inspect module {module_name}: {e}")

