#!/usr/bin/env python3

import logging
logging.basicConfig(level=logging.INFO)

from typing import (
    Any, Dict, Generator, Generic, List, Literal, Optional, Protocol, Set,
    Tuple, TypeVar, Union
)

from functools import lru_cache
from lemminflect import getInflection
from spacy import Language
from spacy.tokens import Span, Doc, Token
from spacy.matcher import Matcher

from .consts import (
    ClauseTypeUndefined, ClauseTypeSV, ClauseTypeSVA, ClauseTypeSVC,
    ClauseTypeSVO, ClauseTypeSVOA, ClauseTypeSVOC, ClauseTypeSVOO,
    ComplementBe, dictionary, verb_phrase_rules
)
from .custom_types import PropositionType, PropositionTypes

# ?? Should these move to func below?
Doc.set_extension("clauses", default=[], force=True)
Span.set_extension("clauses", default=[], force=True)


@Language.component("claucy")
def extract_clauses_doc(doc:Doc) -> Doc:
    # Doc.set_extension("clauses", default=[], force=True)
    # Span.set_extension("clauses", default=[], force=True)
    sent: Span
    for sent in doc.sents:
        clauses = list(extract_clauses(sent))
        sent._.clauses = clauses
        doc._.clauses += clauses
    return doc


def add_to_pipe(nlp: Language) -> None:
    nlp.add_pipe("claucy")


class Clause:
    def __init__(
            self,
            subject:         Optional[Span]        = None,
            verb:            Optional[Span]        = None,
            indirect_object: Optional[Span]        = None,
            direct_object:   Optional[Span]        = None,
            complement:      Optional[Span]        = None,
            adverbials:      Optional[List[Span]]  = [],
    ) -> None:
        """

        Parameters
        ----------
        subject : Span
            Subject.
        verb : Span
            Verb.
        indirect_object : Span, optional
            Indirect object, The default is None.
        direct_object : Span, optional
            Direct object. The default is None.
        complement : Span, optional
            Complement. The default is None.
        adverbials : list, optional
            List of adverbials. The default is [].

        Returns
        -------
        None.

        """

        self.subject:         Optional[Span]       = subject
        self.verb:            Optional[Span]       = verb
        self.indirect_object: Optional[Span]       = indirect_object
        self.direct_object:   Optional[Span]       = direct_object
        self.complement:      Optional[Span]       = complement
        self.adverbials:      Optional[List[Span]] = adverbials if adverbials else []

        self.doc: Optional[Doc] = None
        if self.subject and hasattr(self.subject,"doc"):
            self.doc = self.subject.doc

        self.type: str = self._get_clause_type()


    def _get_clause_type(self,conservative:bool=False) -> str:
        has_verb: bool = bool(self.verb != None)
        has_complement: bool = bool(self.complement != None)
        has_adverbial: bool = bool(
            len(self.adverbials) > 0
            if self.adverbials
            else False
        )

        has_ext_copular_verb: bool = False
        has_non_ext_copular_verb: bool = False
        complex_transitive: bool = False

        if self.verb \
           and hasattr(self.verb,"root") and self.verb.root \
           and hasattr(self.verb.root,"lemma_") and self.verb.root.lemma_:
            has_ext_copular_verb = bool(
                has_verb
                and self.verb.root.lemma_ in dictionary["ext_copular"]
            )
            has_non_ext_copular_verb = bool(
                has_verb
                and self.verb.root.lemma_ in dictionary["non_ext_copular"]
            )
            complex_transitive = (
                has_verb
                and self.verb.root.lemma_ in dictionary["complex_transitive"]
            )

        has_direct_object: bool = bool(self.direct_object != None)
        has_indirect_object: bool = bool(self.indirect_object != None)
        has_object: bool = has_direct_object or has_indirect_object

        clause_type: str = ClauseTypeUndefined

        if not has_verb:
            clause_type = ClauseTypeSVC
            return clause_type

        if has_object:
            if has_direct_object and has_indirect_object:
                clause_type = ClauseTypeSVOO
            elif has_complement:
                clause_type = ClauseTypeSVOC
            elif not has_adverbial or not has_direct_object:
                clause_type = ClauseTypeSVO
            elif complex_transitive or conservative:
                clause_type = ClauseTypeSVOA
            else:
                clause_type = ClauseTypeSVO
        else:
            if has_complement:
                clause_type = ClauseTypeSVC
            elif not has_adverbial or has_non_ext_copular_verb:
                clause_type = ClauseTypeSV
            elif has_ext_copular_verb or conservative:
                clause_type = ClauseTypeSVA
            else:
                clause_type = ClauseTypeSV

        return clause_type


    def __repr__(self) -> str:
        return "<{}, {}, {}, {}, {}, {}, {}>".format(
            self.type,
            self.subject,
            self.verb,
            self.indirect_object,
            self.direct_object,
            self.complement,
            self.adverbials,
        )


    @property
    def propositions(self) -> Generator[Union[PropositionTypes,str], None, None]:
        p: Union[PropositionTypes,str]
        for p in self.get_propositions(inflection=False):
            yield p


    def get_propositions(
            self,
            as_text: bool = False,
            inflection: Optional[Union[str,Literal[False]]] = "VBD",
            capitalize: bool = False
    ) -> Generator[Union[PropositionTypes,str],None,None]:
        if inflection not in ["",False] and not as_text:
            logging.warning(
                "`inflection' argument is ignored when `as_text==False'. "
                "To suppress this warning call `propositions' with the argument `inflection=None'"
            )

        if capitalize and not as_text:
            logging.warning(
                "`capitalize' argument is ignored when `as_text==False'. "
                "To suppress this warning call `propositions' with the argument `capitalize=False"
            )

        subjects:         List[Span] = extract_ccs_from_token_at_root(self.subject)
        direct_objects:   List[Span] = extract_ccs_from_token_at_root(self.direct_object)
        indirect_objects: List[Span] = extract_ccs_from_token_at_root(self.indirect_object)
        complements:      List[Span] = extract_ccs_from_token_at_root(self.complement)
        verbs:            List[Span] = [self.verb] if self.verb else []

        seen: Set[PropositionTypes] = set()

        @lru_cache(typed=True,maxsize=10000)
        def should_yield(i:PropositionTypes) -> bool:
            if i not in seen:
                seen.add(i)
                return True
            return False

        ret: PropositionTypes
        subj: Span

        for subj in subjects:
            if complements and not verbs:
                complement: Span
                for complement in complements:
                    ret = tuple([subj, ComplementBe, complement])
                    if should_yield(ret):
                        if as_text:
                            yield from _convert_clauses_to_text(ret,inflection,capitalize)
                        else:
                            yield ret
                ret = tuple([subj, ComplementBe]) + tuple(complements)
                if should_yield(ret):
                    if as_text:
                        yield from _convert_clauses_to_text(ret,inflection,capitalize)
                    else:
                        yield ret

            verb: Span
            for verb in verbs:
                prop: List[Span] = [subj, verb]
                if self.type in [ClauseTypeSV, ClauseTypeSVA]:
                    if self.adverbials:
                        a1: Span
                        for a1 in self.adverbials:
                            ret = tuple(prop + [a1])
                            if should_yield(ret):
                                if as_text:
                                    yield from _convert_clauses_to_text(ret,inflection,capitalize)
                                else:
                                    yield ret
                        ret = tuple(prop + self.adverbials)
                        if should_yield(ret):
                            if as_text:
                                yield from _convert_clauses_to_text(ret,inflection,capitalize)
                            else:
                                yield ret
                    else:
                        ret = tuple(prop)
                        if should_yield(ret):
                            if as_text:
                                yield from _convert_clauses_to_text(ret,inflection,capitalize)
                            else:
                                yield ret
                elif self.type == ClauseTypeSVOO:
                    iobj: Span
                    dobj: Span
                    for iobj in indirect_objects:
                        for dobj in direct_objects:
                            ret = tuple([subj, verb, iobj, dobj])
                            if should_yield(ret):
                                if as_text:
                                    yield from _convert_clauses_to_text(ret,inflection,capitalize)
                                else:
                                    yield ret
                elif self.type == ClauseTypeSVO:
                    svo_obj: Span
                    for svo_obj in direct_objects + indirect_objects:
                        ret = tuple([subj, verb, svo_obj])
                        if should_yield(ret):
                            if as_text:
                                yield from _convert_clauses_to_text(ret,inflection,capitalize)
                            else:
                                yield ret
                        if self.adverbials:
                            svo_advb: Span
                            for svo_advb in self.adverbials:
                                ret = tuple([subj, verb, svo_obj, svo_advb])
                                if should_yield(ret):
                                    if as_text:
                                        yield from _convert_clauses_to_text(ret,inflection,capitalize)
                                    else:
                                        yield ret
                elif self.type == ClauseTypeSVOA:
                    svoa_obj: Span
                    for svoa_obj in direct_objects:
                        if self.adverbials:
                            svoa_advb: Span
                            for svoa_advb in self.adverbials:
                                ret = tuple(prop + [svoa_obj, svoa_advb])
                                if should_yield(ret):
                                    if as_text:
                                        yield from _convert_clauses_to_text(ret,inflection,capitalize)
                                    else:
                                        yield ret
                            ret = tuple(prop + [svoa_obj] + self.adverbials)
                            if should_yield(ret):
                                if as_text:
                                    yield from _convert_clauses_to_text(ret,inflection,capitalize)
                                else:
                                    yield ret

                elif self.type == ClauseTypeSVOC:
                    svoc_obj: Span
                    for svoc_obj in indirect_objects + direct_objects:
                        if complements:
                            for complement in complements:
                                ret = tuple(prop + [svoc_obj, complement])
                                if should_yield(ret):
                                    if as_text:
                                        yield from _convert_clauses_to_text(ret,inflection,capitalize)
                                    else:
                                        yield ret
                            ret = tuple(prop + [svoc_obj] + complements)
                            if should_yield(ret):
                                if as_text:
                                    yield from _convert_clauses_to_text(ret,inflection,capitalize)
                                else:
                                    yield ret
                elif self.type == ClauseTypeSVC:
                     if complements:
                        for complement in complements:
                            ret = tuple(prop + [complement])
                            if should_yield(ret):
                                if as_text:
                                    yield from _convert_clauses_to_text(ret,inflection,capitalize)
                                else:
                                    yield ret
                        ret = tuple(prop + complements)
                        if should_yield(ret):
                            if as_text:
                                yield from _convert_clauses_to_text(ret,inflection,capitalize)
                            else:
                                yield ret


@lru_cache(typed=True,maxsize=10000)
def inflect_token(token: Token, inflection: Optional[Union[str,Literal[False]]] = "VBD",) -> str:
    tt: Token
    if (
            inflection
            and token.pos_ == "VERB"
            and "AUX" not in [tt.pos_ for tt in token.lefts]
            # t is not preceded by an auxiliary verb (e.g. `the birds were ailing`)
            and token.dep_ != "pcomp"
    ):  # t `dreamed of becoming a dancer`
        return str(token._.inflect(inflection))
    else:
        return str(token)


def _convert_clauses_to_text(
        propositions: PropositionTypes,
        inflection: Optional[Union[str,Literal[False]]] = "VBD",
        capitalize: Optional[bool] = False
) -> Generator[str, None, None]:
    proposition_texts: List[str] = []
    prop: Union[Span,str]  ## mypy complains if PropositionType used here. Why? FIXME
    for prop in propositions:
        span_texts: List[str] = []
        i: Union[Span,Token,str]
        for i in prop:
            if not isinstance(i,Span):
                continue
            token: Token
            span_texts.append(
                " ".join(
                    inflect_token(token, inflection) for token in i
                )
            )
        if capitalize == False:
            yield " ".join(span_texts)
        else:
            yield " ".join(span_texts).capitalize() + "."


@lru_cache(typed=True,maxsize=10000)
def _get_verb_matches(span:Span) -> List[Tuple[int,int,int]]:
    # 1. Find verb phrases in the span
    # (see mdmjsh answer here: https://stackoverflow.com/questions/47856247/extract-verb-phrases-using-spacy)
    verb_matcher: Matcher = Matcher(span.vocab)
    n: str
    v: List[List[Dict[str,str]]]
    for n,v in verb_phrase_rules:
        verb_matcher.add(n,v)
    return verb_matcher(span)


def _get_verb_chunks(span:Span) -> Generator[Span, None, None]:
    # Filter matches (e.g. do not have both "has won" and "won" in verbs)
    seen_root: Set[Token] = set()
    match: Span
    for match in (
            span[start:end]
            for _, start, end
            in _get_verb_matches(span)
    ):
        if match.root not in seen_root:
            seen_root.add(match.root)
            yield match


@lru_cache(typed=True,maxsize=10000)
def _get_subject(verb: Span) -> Union[None,Span]:
    root: Token = verb.root
    flag: bool = False
    while not flag:
        # Is the subject at current level?
        c: Token
        for c in root.children:
            if c.dep_ in ["nsubj", "nsubjpass"]:
                subject: Span = extract_span_from_entity(c)
                return subject
        # If not, move up one level.
        if root.dep_ in ["conj", "cc", "advcl", "acl", "ccomp"] \
           and root != root.head:
            root = root.head
        else:
            flag = True
    return None


@lru_cache(typed=True,maxsize=10000)
def _find_matching_child(root: Token, allowed_types: Tuple[str, ...]) -> Optional[Span]:
    c: Token
    cc: Token
    # This only seems to ever have one item (or perhaps only interested in the first??)
    for cc in ( c for c in root.children if c.dep_ in allowed_types ):
        return extract_span_from_entity(cc)
    return None


def extract_clauses(span: Span) -> Generator[Optional[Clause], None, None]:
    verb: Span
    for verb in _get_verb_chunks(span):
        
        subject: Union[Span,Literal[None]] = _get_subject(verb)
        if not subject:
            continue
        
        # Check if there are phrases of the form, "AE, a scientist of ..."
        # If so, add a new clause of the form:
        # <AE, is, a scientist>
        c: Token
        for c in subject.root.children:
            if c.dep_ == "appos":
                yield Clause(
                    subject=subject,
                    complement=extract_span_from_entity(c)
                )

        indirect_object = _find_matching_child(verb.root, tuple(["dative"]))
        direct_object = _find_matching_child(verb.root, tuple(["dobj"]))
        complement = _find_matching_child(
            verb.root, tuple(["ccomp", "acomp", "xcomp", "attr"])
        )
        adverbials = [
            extract_span_from_entity(c)
            for c in verb.root.children
            if c.dep_ in ("prep", "advmod", "agent")
        ]

        yield Clause(
            subject=subject,
            verb=verb,
            indirect_object=indirect_object,
            direct_object=direct_object,
            complement=complement,
            adverbials=adverbials,
        )


@lru_cache(typed=True,maxsize=10000)
def extract_span_from_entity(token: Token) -> Span:
    return Span(token.doc, start=min(token.subtree).i, end=max(token.subtree).i + 1)


@lru_cache(typed=True,maxsize=10000)
def extract_span_from_entity_no_cc(token: Token) -> Span:
    c: Token
    ent_subtree: List[Token] = sorted(
        [token] + [
            c for c in token.children if c.dep_ not in ["cc", "conj", "prep"]
        ]
    )
    return Span(token.doc, start=min(ent_subtree).i, end=max(ent_subtree).i + 1)


def extract_ccs_from_entity(token: Token) -> Generator[Span, None, None]:
    yield extract_span_from_entity_no_cc(token)
    c: Token
    for c in token.children:
        if c.dep_ in ["conj", "cc"]:
            yield from extract_ccs_from_entity(c)


def extract_ccs_from_token_at_root(span: Union[Span,None]) -> List[Span]:
    if span is None:
        return []
    else:
        return list(extract_ccs_from_token(span.root))


def extract_ccs_from_token(token: Token) -> Generator[Span, None, None]:
    if token.pos_ in ["NOUN", "PROPN", "ADJ"]:
        c: Token
        children = sorted(
            [token] + [
                c for c in token.children
                if c.dep_ in ["advmod", "amod", "det", "poss", "compound"]
            ]
        )
        yield Span(token.doc, start=min(children).i, end=max(children).i + 1)
    else:
        yield Span(token.doc, start=token.i, end=token.i + 1)
    for c in token.children:
        if c.dep_ == "conj":
            yield from extract_ccs_from_token(c)


# if __name__ == "__main__":
#     import spacy
# 
#     nlp = spacy.load("en_core_web_sm")
#     add_to_pipe(nlp)
# 
#     doc = nlp(
#         # "Chester is a banker by trade, but is dreaming of becoming a great dancer."
#         " A cat , hearing that the birds in a certain aviary were ailing dressed himself up as a physician , and , taking his cane and a bag of instruments becoming his profession , went to call on them ."
#     )
# 
#     print(doc._.clauses)
#     for clause in doc._.clauses:
#         print(clause.to_propositions(as_text=True, capitalize=True))
#     print(doc[:].noun_chunks)
