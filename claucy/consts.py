from typing import Dict, List, Tuple

__all__ : List[str] = [
    "ClauseTypeSV",
    "ClauseTypeSVA",
    "ClauseTypeSVC",
    "ClauseTypeSVO",
    "ClauseTypeSVOA",
    "ClauseTypeSVOC",
    "ClauseTypeSVOO",
    "ClauseTypeUndefined",
    "ComplementBe",
    "dictionary",
    "verb_phrase_rules"
]


ClauseTypeSV:        str = "SV"
ClauseTypeSVA:       str = "SVA"
ClauseTypeSVC:       str = "SVC"
ClauseTypeSVO:       str = "SVO"
ClauseTypeSVOA:      str = "SVOA"
ClauseTypeSVOC:      str = "SVOC"
ClauseTypeSVOO:      str = "SVOO"
ClauseTypeUndefined: str = "undefined"


ComplementBe: str = "be"


verb_phrase_rules: List[Tuple[str,List[List[Dict[str,str]]]]] = [
    (
        "Auxiliary verb phrase aux-verb",
        [[ {"POS": "AUX"}, {"POS": "VERB"} ]]
    ),
    (
        "Auxiliary verb phrase",
        [[ {"POS": "AUX"} ]]
    ),
    (
        "Verb phrase",
        [[ {"POS": "VERB"} ]]
    )
]

dictionary: Dict[str,List[str]] = {
    "non_ext_copular": [
        "die", "walk"
    ],
    "ext_copular": [
        "act", "appear",
        "be", "become",
        "come", "come out",
        "do",
        "end up",
        "fall", "feel",
        "get", "go", "grow",
        "keep",
        "leave", "lie", "live", "look", "love",
        "prove",
        "remain",
        "seem", "smell", "sound", "stand", "stay",
        "taste", "try", "turn", "turn up",
        "wind up"
    ],
    "complex_transitive": [
        "bring",
        "catch",
        "drive",
        "get",
        "keep",
        "lay", "lead",
        "place", "put",
        "set", "show", "sit", "slip", "stand",
        "take"
    ],
    "adverbs_ignore": [
        "as",
        "even",
        "so",
        "then", "thus",
        "why"
    ],
    "adverbs_include": [
        "barely",
        "hardly",
        "rarely",
        "scarcely", "seldom"
    ]
}
