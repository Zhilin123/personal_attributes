#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FbDeprecatedDialogTeacher, YamlTeacher
from parlai.utils.data import DatatypeHelper
from parlai.utils.misc import warn_once
from .build import build
from parlai.utils.strings import normalize_reply
import parlai.utils.logging as logging
from parlai.core.params import ParlaiParser
from typing import Optional
from parlai.core.opt import Opt

import copy
import os
import json
from collections import defaultdict


'''All teachers have a version with and without label candidates. Each teacher
defaults to using a dataset with label candidates. To use a dataset without
label candidates, specify this using the task flag:

--task convai2:{TEACHER_NAME}:no_cands

where TEACHER_NAME is None, SelfOriginal (Self), or SelfRevised.
'''

filename = "../parlai_internal/tasks/convai2-rev/sentence_to_triple.json"

file = open(filename, "r")
sentence_to_triple = json.load(file)
new_sentence_to_triple = defaultdict(list)
correct_counter = 0
total_counter = 0


corrections = [("iam", "i'm"), ("ihave", "i've"), ("donot", "don't"), ("iwill", "i'll"), ("hasnot", "hasn't")]

def regular_normalize(text):
    return '\n'.join([normalize_reply(i) for i in text.split('\n')])

def apply_corrections(sentence):
    for correction in corrections:
        sentence = sentence.replace(correction[0], correction[1])
    return sentence

for i in sentence_to_triple:
    key = ''.join(i.split()).lower()
    key = apply_corrections(key)
    new_sentence_to_triple[key] = sentence_to_triple[i]

def change_x(x):
    triple = new_sentence_to_triple[apply_corrections(''.join(x.split()).lower())]
    triple_formatted = " ".join([str(i) for i in triple]).replace("_", " ").replace("[", " ").replace("]", " ") + '.' if triple else ''
    x = triple_formatted + x
    return x

def _path(opt, persona, use_cands):
    # Build the data if it doesn't exist.
    build(opt)
    datatype = opt['datatype'].split(':')[0]
    if datatype == 'test':
        warn_once("WARNING: Test set not included. Setting datatype to valid.")
        datatype = 'valid'
    dt = datatype + '_' + persona
    cands = '' if use_cands else '_no_cands'
    return os.path.join(opt['datapath'], 'ConvAI2', dt + cands + '.txt')


class BothTeacher(FbDeprecatedDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except Exception:
            use_cands = True
        opt['datafile'] = _path(opt, 'both_original', use_cands)
        super().__init__(opt, shared)


class NoneTeacher(FbDeprecatedDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except Exception:
            use_cands = True
        opt['datafile'] = _path(opt, 'none_original', use_cands)
        super().__init__(opt, shared)


class SelfOriginalTeacher(FbDeprecatedDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except Exception:
            use_cands = True
        opt['datafile'] = _path(opt, 'self_original', use_cands)
        super().__init__(opt, shared)


class SelfTeacher(SelfOriginalTeacher):
    pass


class SelfRevisedTeacher(FbDeprecatedDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except Exception:
            use_cands = True
        opt['datafile'] = _path(opt, 'self_revised', use_cands)
        super().__init__(opt, shared)


class NormalizedTeacherTrait(object):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('NormalizedTeacher arguments')
        agent.add_argument(
            '--your-persona-first',
            type='bool',
            default=True,
            help="whether to prepend your persona followed by partner's persona. True by default to be consistent with the BothTeach",
        )
        agent.add_argument(
            '--max-num-turns',
            type=int,
            default=-1,
            help="first X turns per episode to show. If -1 then the whole episode is shown",
        )
        return agent

    def __init__(self, opt, shared=None):
        self.max_num_turns = opt["max_num_turns"]
        self.your_persona_first = opt["your_persona_first"]
        super().__init__(opt, shared)

    def normalize_replies(self, x):

        xs = x.split('\n')
        your_personas = []
        partner_personas = []
        non_personas = []
        for x in xs:
            if x.startswith('your persona: '):
                # Normalize the sentence appearing after 'your persona:'
                x = x[len('your persona: ') :]



                x =change_x(x)

                x = normalize_reply(x)
                x = 'your persona: ' + x
                your_personas.append(x)
            elif x.startswith("partner's persona: "):

                x = x[len("partner's persona: ") :]
                x = normalize_reply(x)
                x = "partner's persona: " + x
                partner_personas.append(x)
            else:

                x =change_x(x)
                x = normalize_reply(x)
                non_personas.append(x)

        xs2 = []
        if self.your_persona_first:
            xs2.extend(your_personas)
            xs2.extend(partner_personas)
        else:
            xs2.extend(partner_personas)
            xs2.extend(your_personas)
        xs2.extend(non_personas)
        #raise ValueError
        return '\n'.join(xs2)

    def setup_data(self, path):
        logging.info(f"loading normalized fbdialog data: {path}")
        exs_counter = 0
        for (text, labels, reward, candidates), new_episode in super().setup_data(path):
            if new_episode:
                exs_counter = 0
            if self.max_num_turns > 0 and exs_counter >= self.max_num_turns:
                continue

            if exs_counter != 0:
                text = regular_normalize(text)
            elif exs_counter == 0:
                normal_text = regular_normalize(text).split('\n')
                transformed_text = self.normalize_replies(text).split('\n')
                text = '\n'.join(transformed_text[:-1] + [normal_text[-1]])
            else:
                text = self.normalize_replies(text)


            labels = [self.normalize_replies(l) for l in labels]
            candidates = [self.normalize_replies(c) for c in candidates]
            exs_counter += 1
            ### debug code
#            print(exs_counter, text)
#            print("labels: ",labels)
#
#            if exs_counter == 5:
#                raise ValueError
            ###

            yield (text, labels, reward, candidates), new_episode


class NormalizedTeacher(NormalizedTeacherTrait, SelfOriginalTeacher):
    pass

class NormalizedRevisedTeacher(NormalizedTeacherTrait, SelfRevisedTeacher):
    pass

class NormalizedBothTeacher(NormalizedTeacherTrait, BothTeacher):
    pass


class NormalizedTheirTeacher(NormalizedTeacherTrait, BothTeacher):
    def normalize_replies(self, x):
        xs = x.split('\n')
        xs2 = []
        for x in xs:
            if x.startswith('your persona: '):
                continue
            elif x.startswith("partner's persona: "):
                x = x[len("partner's persona: ") :]
                x = normalize_reply(x)
                x = "partner's persona: " + x
            else:
                x = normalize_reply(x)
            xs2.append(x)
        return '\n'.join(xs2)


class NormalizedNoneTeacher(NormalizedTeacherTrait, NoneTeacher):
    pass


class DefaultTeacher(SelfOriginalTeacher):
    pass


class InteractiveTeacher(SelfOriginalTeacher):
    # Dummy class to add arguments for interactive world.
    pass


class SelfchatTeacher(SelfOriginalTeacher):
    # Dummy class to add arguments for interactive world.
    pass


class SampleTeacher(YamlTeacher):
    """
    Loads the small sample of data created by the AutoTeacherTests.
    """

    def __init__(self, opt, shared=None):
        opt = opt.copy()
        fold = DatatypeHelper.fold(opt['datatype'])
        opt['datafile'] = os.path.join(
            os.path.dirname(__file__), f'test/convai2_{fold}.yml'
        )
        super().__init__(opt, shared)
