import inspect
from gradio import TabbedInterface

from synthetic_dataset_generator import (  # noqa
    _distiset,
    _inference_endpoints,
)
from synthetic_dataset_generator.load_env import init_environment

# Initialize environment variables
init_environment()


def launch(*args, **kwargs):
    """Launch the synthetic dataset generator.
    Based on the `TabbedInterface` from Gradio.
    Parameters: https://www.gradio.app/docs/gradio/tabbedinterface
    """
    from synthetic_dataset_generator.app import demo
    return demo.launch(*args, server_name="0.0.0.0", **kwargs)


launch.__doc__ = TabbedInterface.launch.__doc__
launch.__signature__ = inspect.signature(TabbedInterface.launch)
launch.__annotations__ = TabbedInterface.launch.__annotations__
