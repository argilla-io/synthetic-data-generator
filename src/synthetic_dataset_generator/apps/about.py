import gradio as gr

with gr.Blocks() as app:
    gr.Markdown(
        """
        Synthetic data is artificially generated information that mimics real-world data. It allows overcoming data limitations by expanding or enhancing datasets.

        Introducing the Synthetic Data Generator, a user-friendly application that takes a no-code approach to creating custom datasets with Large Language Models (LLMs). The best part: A simple step-by-step process, making dataset creation a non-technical breeze, allowing anyone to create datasets and models in minutes and without any code.

        The synthetic data generator takes your custom prompt and returns a dataset for your use case, using a synthetic data pipeline. In the background this is powered by [distilabel](https://distilabel.argilla.io/latest/) and the [free Hugging Face text-generation API](https://huggingface.co/docs/api-inference/en/index) but we don't need to worry about these complexities and we can focus on using the UI.

        - Read more in [our announcement blog post](https://huggingface.co/blog/synthetic-data-generator)
        - Find the library on [GitHub](https://github.com/argilla-io/synthetic-data-generator)
        """
    )
