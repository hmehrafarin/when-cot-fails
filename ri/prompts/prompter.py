from typing import List, Optional
import json
import os.path as osp

# Default templates directory is relative to this file
_DEFAULT_TEMPLATES_DIR = osp.join(osp.dirname(__file__), "templates")


class Prompter:
    """
    Manages prompt templates for experiments.

    Loads a JSON template file and constructs prompts for query batches.
    """

    __slots__ = (
        "template",
        "_verbose",
        "_start_context",
        "_end_context",
        "template_name",
    )

    def __init__(
        self,
        template_name: str = "",
        templates_dir: Optional[str] = None,
    ):
        """
        Initialize the prompter with a template.

        Parameters
        ----------
        template_name : str
            Name of the template file (without .json extension).
        templates_dir : str | None
            Directory containing template files. Defaults to the package's
            built-in templates directory.
        """
        if templates_dir is None:
            templates_dir = _DEFAULT_TEMPLATES_DIR
        if not template_name:
            template_name = "gsm8k"
        self.template_name = template_name
        file_name = osp.join(templates_dir, f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
            self._start_context = self.template["start_of_context"]
            self._end_context = self.template["end_of_context"]

    @property
    def start_context(self) -> str:
        return self._start_context

    @property
    def end_context(self) -> str:
        return self._end_context

    def create_query_prompts(self, batch_row: List[dict]) -> List[str]:
        """
        Returns the list of query prompts formatted for each item in batch_row.
        """
        queries = []
        for row in batch_row:
            query = self.template["query"].format(
                question=row["question"], answer=row["answer"]
            ).strip()
            queries.append(query)
        return queries

    def create_prompt(
        self,
        batch_row: List[dict],
    ) -> List[str]:
        """
        Creates a batch of prompts from query data.

        Parameters
        ----------
        batch_row : List[dict]
            List of samples with 'question' and 'answer' keys.

        Returns
        -------
        List[str]
            Formatted prompts.
        """
        return self.create_query_prompts(batch_row)

    def get_response(self, output: str) -> str:
        """Extract the response portion from model output."""
        response_split = self.template["response_split"]
        output = output.split(response_split)[-1].strip()
        return output
