import warnings

import distilabel
import distilabel.distiset
from distilabel.models import InferenceEndpointsLLM
from pydantic import (
    ValidationError,
    model_validator,
)


class CustomInferenceEndpointsLLM(InferenceEndpointsLLM):
    @model_validator(mode="after")  # type: ignore
    def only_one_of_model_id_endpoint_name_or_base_url_provided(
        self,
    ) -> "InferenceEndpointsLLM":
        """Validates that only one of `model_id` or `endpoint_name` is provided; and if `base_url` is also
        provided, a warning will be shown informing the user that the provided `base_url` will be ignored in
        favour of the dynamically calculated one.."""

        if self.base_url and (self.model_id or self.endpoint_name):
            warnings.warn(  # type: ignore
                f"Since the `base_url={self.base_url}` is available and either one of `model_id`"
                " or `endpoint_name` is also provided, the `base_url` will either be ignored"
                " or overwritten with the one generated from either of those args, for serverless"
                " or dedicated inference endpoints, respectively."
            )

        if self.use_magpie_template and self.tokenizer_id is None:
            raise ValueError(
                "`use_magpie_template` cannot be `True` if `tokenizer_id` is `None`. Please,"
                " set a `tokenizer_id` and try again."
            )

        if (
            self.model_id
            and self.tokenizer_id is None
            and self.structured_output is not None
        ):
            self.tokenizer_id = self.model_id

        if self.base_url and not (self.model_id or self.endpoint_name):
            return self

        if self.model_id and not self.endpoint_name:
            return self

        if self.endpoint_name and not self.model_id:
            return self

        raise ValidationError(
            f"Only one of `model_id` or `endpoint_name` must be provided. If `base_url` is"
            f" provided too, it will be overwritten instead. Found `model_id`={self.model_id},"
            f" `endpoint_name`={self.endpoint_name}, and `base_url`={self.base_url}."
        )


distilabel.models.llms.InferenceEndpointsLLM = CustomInferenceEndpointsLLM
