from fastenum import Enum as FEnum


class ClauseType(FEnum):
    undefined: None = None
    SV:        str  = "SV"
    SVA:       str  = "SVA"
    SVC:       str  = "SVC"
    SVO:       str  = "SVO"
    SVOA:      str  = "SVOA"
    SVOC:      str  = "SVOC"
    SVOO:      str  = "SVOO"

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return str(self.name)
