from typing import List

__all__ : List[str] = [
    "ClauseType",
    "Complement"
]


class ClauseType:
    undefined: None = None
    SV:        str  = "SV"
    SVA:       str  = "SVA"
    SVC:       str  = "SVC"
    SVO:       str  = "SVO"
    SVOA:      str  = "SVOA"
    SVOC:      str  = "SVOC"
    SVOO:      str  = "SVOO"


class Complement:
    be: str = "be"
