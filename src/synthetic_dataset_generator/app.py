from synthetic_dataset_generator._tabbedinterface import TabbedInterface

# from synthetic_dataset_generator.apps.eval import app as eval_app
from synthetic_dataset_generator.apps.readme import app as readme_app
from synthetic_dataset_generator.apps.sft import app as sft_app
from synthetic_dataset_generator.apps.textcat import app as textcat_app

theme = "argilla/argilla-theme"

css = """
.main_ui_logged_out{opacity: 0.3; pointer-events: none}
button[role="tab"][aria-selected="true"] { border: 0; background: var(--button-primary-background-fill); color: white; border-top-right-radius: var(--radius-md); border-top-left-radius: var(--radius-md)}
button[role="tab"][aria-selected="true"]:hover {border-color: var(--button-primary-background-fill); background: var(var(--button-primary-background-fill-hover))}
.tabitem { border: 0; padding-inline: 0}
.gallery-item {background: var(--background-fill-secondary); text-align: left}
#system_prompt_examples { color: var(--body-text-color) !important; background-color: var(--block-background-fill) !important;}
.container {padding-inline: 0 !important}
#sign_in_button { flex-grow: 0; width: auto !important; display: flex; align-items: center; justify-content: center; margin: 0 auto; }
"""

image = """<br><img src="https://raw.githubusercontent.com/argilla-io/synthetic-data-generator/main/assets/logo.svg" alt="Synthetic Data Generator Logo" style="display: block; margin-left: auto; margin-right: auto; width: clamp(50%, 400px, 100%)"/>"""

demo = TabbedInterface(
    [textcat_app, sft_app, readme_app],
    ["Text Classification", "Supervised Fine-Tuning", "README"],
    css=css,
    title=image,
    head="Synthetic Data Generator",
    theme=theme,
)


if __name__ == "__main__":
    demo.launch()
