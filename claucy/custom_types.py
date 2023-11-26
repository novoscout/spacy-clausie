from typing import Tuple, TypeVar
from spacy.tokens import Span


PropositionType = TypeVar(
    "PropositionType",
    Span,
    str,
)

PropositionTypes = Tuple[PropositionType, ...]
