""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''

import re
import sys
from unidecode import unidecode
import pyopenjtalk
from jamo import h2j, j2hcj
from pypinyin import lazy_pinyin, BOPOMOFO
import jieba, cn2an
from num2words import num2words
import inflect
from phonemizer import phonemize

# This is a list of Korean classifiers preceded by pure Korean numerals.
_korean_classifiers = '군데 권 개 그루 닢 대 두 마리 모 모금 뭇 발 발짝 방 번 벌 보루 살 수 술 시 쌈 움큼 정 짝 채 척 첩 축 켤레 톨 통'

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# Regular expression matching Japanese without punctuation marks:
_japanese_characters = re.compile(r'[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]')

# Regular expression matching non-Japanese characters or punctuation marks:
_japanese_marks = re.compile(r'[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]

# List of (hangul, hangul divided) pairs:
_hangul_divided = [(re.compile('%s' % x[0]), x[1]) for x in [
  ('ㄳ', 'ㄱㅅ'),
  ('ㄵ', 'ㄴㅈ'),
  ('ㄶ', 'ㄴㅎ'),
  ('ㄺ', 'ㄹㄱ'),
  ('ㄻ', 'ㄹㅁ'),
  ('ㄼ', 'ㄹㅂ'),
  ('ㄽ', 'ㄹㅅ'),
  ('ㄾ', 'ㄹㅌ'),
  ('ㄿ', 'ㄹㅍ'),
  ('ㅀ', 'ㄹㅎ'),
  ('ㅄ', 'ㅂㅅ'),
  ('ㅘ', 'ㅗㅏ'),
  ('ㅙ', 'ㅗㅐ'),
  ('ㅚ', 'ㅗㅣ'),
  ('ㅝ', 'ㅜㅓ'),
  ('ㅞ', 'ㅜㅔ'),
  ('ㅟ', 'ㅜㅣ'),
  ('ㅢ', 'ㅡㅣ'),
  ('ㅑ', 'ㅣㅏ'),
  ('ㅒ', 'ㅣㅐ'),
  ('ㅕ', 'ㅣㅓ'),
  ('ㅖ', 'ㅣㅔ'),
  ('ㅛ', 'ㅣㅗ'),
  ('ㅠ', 'ㅣㅜ')
]]

# List of (Latin alphabet, hangul) pairs:
_latin_to_hangul = [(re.compile('%s' % x[0], re.IGNORECASE), x[1]) for x in [
  ('a', '에이'),
  ('b', '비'),
  ('c', '시'),
  ('d', '디'),
  ('e', '이'),
  ('f', '에프'),
  ('g', '지'),
  ('h', '에이치'),
  ('i', '아이'),
  ('j', '제이'),
  ('k', '케이'),
  ('l', '엘'),
  ('m', '엠'),
  ('n', '엔'),
  ('o', '오'),
  ('p', '피'),
  ('q', '큐'),
  ('r', '아르'),
  ('s', '에스'),
  ('t', '티'),
  ('u', '유'),
  ('v', '브이'),
  ('w', '더블유'),
  ('x', '엑스'),
  ('y', '와이'),
  ('z', '제트')
]]

# List of (Latin alphabet, bopomofo) pairs:
_latin_to_bopomofo = [(re.compile('%s' % x[0], re.IGNORECASE), x[1]) for x in [
  ('a', 'ㄟˉ'),
  ('b', 'ㄅㄧˋ'),
  ('c', 'ㄙㄧˉ'),
  ('d', 'ㄉㄧˋ'),
  ('e', 'ㄧˋ'),
  ('f', 'ㄝˊㄈㄨˋ'),
  ('g', 'ㄐㄧˋ'),
  ('h', 'ㄝˇㄑㄩˋ'),
  ('i', 'ㄞˋ'),
  ('j', 'ㄐㄟˋ'),
  ('k', 'ㄎㄟˋ'),
  ('l', 'ㄝˊㄛˋ'),
  ('m', 'ㄝˊㄇㄨˋ'),
  ('n', 'ㄣˉ'),
  ('o', 'ㄡˉ'),
  ('p', 'ㄆㄧˉ'),
  ('q', 'ㄎㄧㄡˉ'),
  ('r', 'ㄚˋ'),
  ('s', 'ㄝˊㄙˋ'),
  ('t', 'ㄊㄧˋ'),
  ('u', 'ㄧㄡˉ'),
  ('v', 'ㄨㄧˉ'),
  ('w', 'ㄉㄚˋㄅㄨˋㄌㄧㄡˋ'),
  ('x', 'ㄝˉㄎㄨˋㄙˋ'),
  ('y', 'ㄨㄞˋ'),
  ('z', 'ㄗㄟˋ')
]]


# List of (bopomofo, romaji) pairs:
_bopomofo_to_romaji = [(re.compile('%s' % x[0], re.IGNORECASE), x[1]) for x in [
  ('ㄅㄛ', 'p⁼wo'),
  ('ㄆㄛ', 'pʰwo'),
  ('ㄇㄛ', 'mwo'),
  ('ㄈㄛ', 'fwo'),
  ('ㄅ', 'p⁼'),
  ('ㄆ', 'pʰ'),
  ('ㄇ', 'm'),
  ('ㄈ', 'f'),
  ('ㄉ', 't⁼'),
  ('ㄊ', 'tʰ'),
  ('ㄋ', 'n'),
  ('ㄌ', 'l'),
  ('ㄍ', 'k⁼'),
  ('ㄎ', 'kʰ'),
  ('ㄏ', 'h'),
  ('ㄐ', 'ʧ⁼'),
  ('ㄑ', 'ʧʰ'),
  ('ㄒ', 'ʃ'),
  ('ㄓ', 'ʦ`⁼'),
  ('ㄔ', 'ʦ`ʰ'),
  ('ㄕ', 's`'),
  ('ㄖ', 'ɹ`'),
  ('ㄗ', 'ʦ⁼'),
  ('ㄘ', 'ʦʰ'),
  ('ㄙ', 's'),
  ('ㄚ', 'a'),
  ('ㄛ', 'o'),
  ('ㄜ', 'ə'),
  ('ㄝ', 'e'),
  ('ㄞ', 'ai'),
  ('ㄟ', 'ei'),
  ('ㄠ', 'au'),
  ('ㄡ', 'ou'),
  ('ㄧㄢ', 'yeNN'),
  ('ㄢ', 'aNN'),
  ('ㄧㄣ', 'iNN'),
  ('ㄣ', 'əNN'),
  ('ㄤ', 'aNg'),
  ('ㄧㄥ', 'iNg'),
  ('ㄨㄥ', 'uNg'),
  ('ㄩㄥ', 'yuNg'),
  ('ㄥ', 'əNg'),
  ('ㄦ', 'əɻ'),
  ('ㄧ', 'i'),
  ('ㄨ', 'u'),
  ('ㄩ', 'ɥ'),
  ('ˉ', '→'),
  ('ˊ', '↑'),
  ('ˇ', '↓↑'),
  ('ˋ', '↓'),
  ('˙', ''),
  ('，', ','),
  ('。', '.'),
  ('！', '!'),
  ('？', '?'),
  ('—', '-')
]]


class GTConfig:
    """
    Generic super class for config settings, see subclasses
    """

    pass  # no specific functionality yet


class GenericReplacementConfig(GTConfig):
    """
    CONFIG for generic settings or replacements/mappings
    """

    # NULL byte for masking an intentional (temporary) separator of items in the text
    # (e.g. for separating currency symbols so that they are not interpreted as abbreviations or acronyms)
    SEP_MASK = '\x00'

    # source for "unicode-to-ascii" mappings (adapted):
    # https://github.com/AASHISHAG/deepspeech-german/blob/master/pre-processing/text_cleaning.py
    UNICODE_TO_ASCII = {
        'àáâãåāăąǟǡǻȁȃȧ': 'a',
        'æǣǽ': 'ä',
        'çćĉċč': 'c',
        'ďđ': 'd',
        'èéêëēĕėęěȅȇȩε': 'e',
        'ĝğġģǥǧǵ': 'g',
        'ĥħȟ': 'h',
        'ìíîïĩīĭįıȉȋ': 'i',
        'ĵǰ': 'j',
        'ķĸǩǩκ': 'k',
        'ĺļľŀł': 'l',
        'м': 'm',
        'ñńņňŉŋǹ': 'n',
        'òóôõōŏőǫǭȍȏðο': 'o',
        'œøǿ': 'ö',
        'ŕŗřȑȓ': 'r',
        'śŝşšș': 's',
        'ţťŧț': 't',
        'ùúûũūŭůűųȕȗ': 'u',
        'ŵ': 'w',
        'ýÿŷ': 'y',
        'źżžȥ': 'z',
    }


class AcronymPhonemeConfig(GTConfig):
    """
    CONFIG for transliterate_op "acronym_phoneme"
    """

    # abbreviations to be excluded since they are regularly pronounced as any other word (pronounceable due to vowels)
    EXCLUDE = {'DAX',
               'NASDAQ',
               'TAZ',
               'WAZ'}

    # when transliterating abbreviations, these mappings for replacing letters are in use
    LETTER = {
        'A': 'ah',
        'B': 'beh',
        'C': 'zee',
        'D': 'deh',
        'E': 'eh',
        'F': 'eff',
        'G': 'geh',
        'H': 'hah',
        'I': 'ih',
        'J': 'jott',
        'K': 'kah',
        'L': 'ell',
        'M': 'emm',
        'N': 'enn',
        'O': 'oh',
        'P': 'peh',
        'Q': 'kuh',
        'R': 'err',
        'S': 'ess',
        'T': 'teh',
        'U': 'uh',
        'V': 'fau',
        'W': 'weh',
        'X': 'iks',
        'Y': 'üpsilon',
        'Z': 'zett',
        'Ä': 'äh',
        'Ö': 'öh',
        'Ü': 'üh',
    }


class UnitConfig(GTConfig):
    """
    CONFIG for units of various physical measures like time, weight, length, area, volume etc.
    """

    # units with specifics: singular equals plural
    # note: '_' is used to mask separator used with abbreviations (see sep_abbreviation=...)
    PLURAL_NO_SUFFIX = {
        'mg': 'milligramm',
        'kg': 'kilogramm',
        'g': 'gramm',
        'nm': 'nanometer',
        'µm': 'mikrometer',
        'mm': 'millimeter',
        'mm^2': 'quadratmillimeter',
        'mm²': 'quadratmillimeter',
        'cm': 'zentimeter',
        'cm^2': 'quadratzentimeter',
        'cm²': 'quadratzentimeter',
        'cm^3': 'kubikzentimeter',
        'cm³': 'kubikzentimeter',
        'dm': 'dezimeter',
        'm': 'meter',
        'm^2': 'quadratmeter',
        'm²': 'quadratmeter',
        'm^3': 'kubikmeter',
        'm³': 'kubikmeter',
        'km': 'kilometer',
        'km^2': 'quadratkilometer',
        'km²': 'quadratkilometer',
        'ha': 'hektar',
        'w': 'watt',
        'j': 'joule',
        'kj': 'kilojoule',
        'k_b': 'kilobyte',
        'm_b': 'megabyte',
        'g_b': 'gigabyte',
        't_b': 'terabyte',
        'p_b': 'petabyte',
        'k_w': 'kilowatt',
        'kb': 'kilobyte',
        'mb': 'megabyte',
        'gb': 'gigabyte',
        'tb': 'terabyte',
        'pb': 'petabyte',
        'kw': 'kilowatt',
        'm_w': 'megawatt',
        'g_w': 'gigawatt',
        'mw': 'megawatt',
        'gw': 'gigawatt',
        '°': 'grad',
        '°c': 'grad celsius',
        '°f': 'grad fahrenheit',
    }

    # units with specifics: plural is singular + suffix 'n'
    PLURAL_SUFFIX_N = {
        't': 'tonnen',
        'kt': 'kilotonnen',
        'mt': 'megatonnen',
        'kwh': 'kilowattstunden',
        'mwh': 'megawattstunden',
        'gwh': 'gigawattstunden',
        'kal': 'kalorien',
        'cal': 'kalorien',
        'mia': 'milliarden',
        'mrd': 'milliarden',
        'md': 'milliarden',
        'brd': 'billiarden',
        'ns': 'nanosekunden',
        'µs': 'mikrosekunden',
        'ms': 'millisekunden',
        's': 'sekunden',
        'sek': 'sekunden',
        'm': 'minuten',
        'min': 'minuten',
        'h': 'stunden',
    }

    PLURAL_SUFFIX_EN = {
        'm': 'millionen',
        'mio': 'millionen',
        'mill': 'millionen',
        'bill': 'billionen',
    }


class AbbreviationConfig(GTConfig):
    """
    CONFIG for general abbreviations including time, weekday, month etc.
    """

    MISC = {
        'fr': 'frau',
        'hr': 'herr',
        'dr': 'doktor',
        'prof': 'professor',
        'jprof': 'juniorprofessor',
        'jun.prof': 'juniorprofessor',
        'mag': 'magister',
        'bsc': 'bachelor of science',
        'msc': 'master of science',
        'st': 'sankt',
        'skt': 'sankt',
        'z.b': 'zum beispiel',
        'bspw': 'beispielsweise',
        'd.h': 'das heißt',
        'abzgl': 'abzüglich',
        'zzgl': 'zuzüglich',
        'ust': 'umsatzsteuer',
        'mwst': 'mehrwertsteuer',
        'ca': 'circa',
        'inkl': 'inklusive',
        'incl': 'inklusive',
        'exkl': 'exklusive',
        'excl': 'exklusive',
        'i.o': 'in ordnung',
        'z.t': 'zum teil',
        'pr': 'pro',
        'ihv': 'in höhe von',
        'i.h.v': 'in höhe von',
        'vglw': 'vergleichsweise'
    }

    TIME = {
        'ns': 'nanosekunden',
        'µs': 'mikrosekunden',
        'ms': 'millisekunden',
        's': 'sekunden',
        'sek': 'sekunden',
        'm': 'minuten',
        'min': 'minuten',
        'h': 'stunden',
    }

    WEEKDAY = {
        'mo': 'montag',
        'di': 'dienstag',
        'mi': 'mittwoch',
        'do': 'donnerstag',
        'fr': 'freitag',
        'sa': 'samstag',
        'so.': 'sonntag',
    }

    MONTH = {
        'januar': 'januar',
        'jänner': 'jänner',
        'februar': 'februar',
        'märz': 'märz',
        'april': 'april',
        'mai': 'mai',
        'juni': 'juni',
        'juli': 'juli',
        'august': 'august',
        'september': 'september',
        'oktober': 'oktober',
        'november': 'november',
        'dezember': 'dezember',
        'jan.': 'januar',
        'jän': 'jänner',
        'feb': 'februar',
        'mrz': 'märz',
        'mär': 'märz',
        'apr': 'april',
        'jun': 'juni',
        'jul': 'juli',
        'aug': 'august',
        'sep': 'september',
        'okt': 'oktober',
        'nov': 'november',
        'dez': 'dezember',
    }

    NUMBER_MONTH = {
        '1': 'januar',
        '01': 'januar',
        '2': 'februar',
        '02': 'februar',
        '3': 'märz',
        '03': 'märz',
        '4': 'april',
        '04': 'april',
        '5': 'mai',
        '05': 'mai',
        '6': 'juni',
        '06': 'juni',
        '7': 'juli',
        '07': 'juli',
        '8': 'august',
        '08': 'august',
        '9': 'september',
        '09': 'september',
        '10': 'oktober',
        '11': 'november',
        '12': 'dezember',
    }

    # '_' is used to mask separator used with abbreviations (see sep_abbreviation=...)
    CURRENCY_SYMBOL = {
        'g_b_p': 'britische pfund',
        '£': 'pfund',
        '$': 'dollar',
        'e_u_r': 'euro',
        't_e_u_r_o': 'tausend euro',
        't_e_u_r': 'tausend euro',
        't€': 'tausend euro',
        '€': 'euro',
        't_d_m': 'tausend d-mark',
        'd_m': 'd-mark',
        'd_k_k': 'dänische kronen',
        's_e_k': 'schwedische kronen',
    }

    # order of items is important: longer strings before substrings, e.g. "<=" before "=" and "<"
    MATH_SYMBOL = {
        '+': ' plus ',
        '-': ' minus ',
        '−': ' minus ',
        '/': ' geteilt durch ',
        '\\': ' modulo ',
        '**': ' hoch ',
        '*': ' mal ',
        '×': ' mal ',
        'x': ' mal ',
        '^': ' hoch ',
        '>=': ' größer gleich ',
        '≥': ' größer gleich ',
        '<=': ' größer gleich ',
        '≤': ' größer gleich ',
        '==': ' äquvivalent zu ',
        '=': ' gleich ',
        '≍': ' äquvivalent zu ',
        '>': ' größer ',
        '<': ' kleiner ',
    }

    # symbols (e.g. punctuations) which are transliterated to spoken words - probably only useful for use in TTS
    # pipelines
    SPOKEN_SYMBOL = {
        ('(', ')'): '_in klammern_',
        ('[', ']'): '_in klammern_',
        ('"', '"'): '_in anführungszeichen_',
        ('\'', '\''): '_zitat_',
    }


class RegExConfig(GTConfig):
    """
    CONFIG for various regular expressions used for transliterations
    """

    def __init__(self, generic_config, acronym_phoneme_config, unit_config, abbreviation_config):
        """
        Constructor

        :param generic_config:
        :param acronym_phoneme_config:
        :param unit_config:
        :param abbreviation_config:
        """
        try:
            self.generic_config = generic_config
            self.acronym_phoneme_config = acronym_phoneme_config
            self.unit_config = unit_config
            self.abbreviation_config = abbreviation_config

            self.SPECIAL_TRANSLITERATE = {
                '.*\+/\-.*': ('+/-', 'plus minus'),
                '.*&.*': ('&', ' und '),
                '(^|(?<=[\.!?;:\-\s]))([a-z]{0,1}|[\d]+)\s{0,1}[^-]\-\s{0,1}([\d]+|[a-z]{0,1})($|(?=[\.!?;:\-\s]+))': (
                    '-', ' bis '),
                # TODO: include more dash operators in unicode...
                '\\b[02-9]+\s{0,1}/\s{0,1}\d+\\b': ('/', ' von '),
                '\\b1/10\\b': ('1/10', 'ein zehntel'),
                '\\b⅒\\b': ('⅒', 'ein zehntel'),
                '\\b1/9\\b': ('1/9', 'ein neuntel'),
                '\\b⅑\\b': ('⅑', 'ein neuntel'),
                '\\b1/8\\b': ('1/8', 'ein achtel'),
                '\\b⅛\\b': ('⅛', 'ein achtel'),
                '\\b1/7\\b': ('1/7', 'ein siebtel'),
                '\\b⅐\\b': ('⅐', 'ein siebtel'),
                '\\b1/6\\b': ('1/6', 'ein sechstel'),
                '\\b⅙\\b': ('⅙', 'ein sechstel'),
                '\\b1/5\\b': ('1/5', 'ein fünftel'),
                '\\b⅕\\b': ('⅕', 'ein fünftel'),
                '\\b1/4\\b': ('1/4', 'ein viertel'),
                '\\b¼\\b': ('¼', 'ein viertel'),
                '\\b1/3\\b': ('1/3', 'ein drittel'),
                '\\b⅓\\b': ('⅓', 'ein drittel'),
                '\\b1/2\\b': ('1/2', 'ein halb'),
                '\\b½\\b': ('½', 'ein halb'),
                '\\b1000\\b': ('1000', 'tausend'),
            }

            self.CURRENCY_MAGNITUDE = ['\\bmia\\b', '\\bmia\.\\b', '\\bmrd\\b', '\\bmrd\.\\b', '\\bmd\\b', '\\bmd\.\\b',
                                       '\\bmilliarde[n]{0,1}\\b', '\\bbrd\\b', '\\bbrd\.\\b', '\\bbilliarde[n]{0,1}\\b',
                                       '\\bmio\\b', '\\bmio\.\\b', '\\bmill\\b', '\\bmill\.\\b',
                                       '\\bmillion(en){0,1}\\b',
                                       '\\bbill\\b', '\\bbill\.\\b', '\\bbillion(en){0,1}\\b', '\\btausend\\b']

            self.DETECT_ABBREVIATION = re.compile(
                '(^|(?<=[\.!?;:\-\s,\(\[\{]))([A-ZÄÖÜ]{2,}|([A-ZÄÖÜ]\.){2,})($|(?=[\.!?;:\-\s,\)\]\}]+))')
            self.DETECT_WEEKDAY = re.compile('\\b(' + '|'.join(self.abbreviation_config.WEEKDAY.keys()) + ')\\b')
            self.DETECT_MONTH = re.compile('\\b(' + '|'.join(self.abbreviation_config.MONTH.keys()) + ')\\b')
            self.DETECT_TIME_OF_DAY = re.compile(
                '(\\b(([0-1][0-9]|2[0-3])[\.\:]|[0-9][\.\:])([0-5][0-9]|[0-9])\s{0,1}(h|uhr){0,1}\\b)')
            self.DETECT_TIMESTAMP = re.compile(
                '(\\b\d+(h|std){0,1}:([0-5][0-9]|[0-9])(m|min){0,1}(:([0-5][0-9]|[0-9])(s|sek|sec){0,1}){0,1}\\b)')
            self.DETECT_DATE = re.compile(
                '(([1-9]|(0(1|2|3|4|5|6|7|8|9))|(1[0-9])|(2[0-9])|30|31)\.(((([1-9]|0(1|2|3|4|5|6|7|8|9))|(10|11|12))\.)|(\s{0,1}('
                + '|'.join(self.abbreviation_config.MONTH.keys()) + ')(\.|\\b)))(\s{0,1}\d\d\d\d|\s{0,1}\d\d){0,1})')
            self.DETECT_ORDINAL = re.compile('[\(\[]{0,1}\d+\.[\)\]]{0,1}')
            self.DETECT_NUMBER = re.compile('([\+\-]{0,1}\d+[\d\.,]*)')
            self.DETECT_WHITESPACE_SEQ = re.compile('\s+')

            # static regex patterns but contents depending on input parameter config (thus set to None, initially)
            self.DETECT_CURRENCY_SYMBOL = None
            self.DETECT_CURRENCY_MAGNITUDE = None
            self.DETECT_CURRENCY = None
        except Exception as e:
            print('', file=sys.stderr)
            print('*** An exception occurred in section', sys._getframe().f_code.co_name,
                  'of class', type(self).__name__, '- see Traceback for details',
                  file=sys.stderr)
            print('', file=sys.stderr)
            raise e


class GermanTransliterate:
    """
    Transliterates *German* text into a normalized form as given by transliter_ops parameter config

    NOTE: This library can be used for cleaning messy text or transliterating/normalizing any text in text mining
    (e.g. with unicode to ascii mappings, various replacements etc.) but in particular its focus is on preprocessing
    text in TTS training or inference pipelines (applications in ASR may be useful, too)
    """

    def __init__(self,
                 transliterate_ops=['acronym_phoneme', 'accent_peculiarity', 'amount_money', 'date', 'timestamp',
                                    'time_of_day', 'ordinal', 'special', 'spoken_symbol'],
                 replace={'-': ' '},
                 sep_abbreviation=' ',
                 make_lowercase=True
                 ):
        """
        Constructor

          USE WITH PHONEMES

          For phonemic use of the result, for instance as direct input to espeak-ng in order to get espeak or
          IPA encodings from the text, use these settings:
            replace={';': ',', ':': ' '},
            sep_abbreviation=' -- '


        :param transliterate_ops: list of keywords to signal which transliterate operations should be carried out (see documentation for details of keyword meaning!)
        :param replace: dict of "original: replacement" string tuples to be used as additional plain and simple "on-the-fly" replacements with the text, e.g replace={'-' : ' '} replaces all dashes with whitespace
        :param sep_abbreviation: a special separator used for transliteration of abbreviations; this is mostly only useful with phonemic encoding of a text as a next step in a TTS pipeline
        :param make_lowercase: if True, text is made lowercase (default), NOTE: most of the transliterate operations do *only* work with make_lowercase=True - this is due to the various dictionaries operating with lowercase only. Please use make_lowercase=False *only* when transliterate_ops aren't overly used, otherwise most of them do not work!
        """
        try:
            self.transliterate_ops = transliterate_ops
            self.replace = replace
            self.sep_abbreviation = sep_abbreviation
            self.make_lowercase = make_lowercase

            # config instantiations
            self.generic_config = GenericReplacementConfig()
            self.acronym_phoneme_config = AcronymPhonemeConfig()
            self.unit_config = UnitConfig()
            self.abbreviation_config = AbbreviationConfig()
            self.regex = RegExConfig(self.generic_config,
                                     self.acronym_phoneme_config,
                                     self.unit_config,
                                     self.abbreviation_config)

            # prepare and configure static strings with regular expressions (depending on input config parameters)
            escaped_cursym = [re.escape(it) for it in self.abbreviation_config.CURRENCY_SYMBOL.keys()]
            rstring_cursym_escaped = '|'.join(escaped_cursym).replace('_', self.generic_config.SEP_MASK)
            rstring_curmagn = '|'.join(self.regex.CURRENCY_MAGNITUDE)
            self.regex.DETECT_CURRENCY_SYMBOL = re.compile(rstring_cursym_escaped)
            self.regex.DETECT_CURRENCY_MAGNITUDE = re.compile(rstring_curmagn)

            cur_str = '(^|(?<=[\.!?;:\-\(\)\[\]\s]))(([\+\-]{0,1}\d+[\d\.,]*\s*(' \
                        + rstring_curmagn + '){0,1}\s*' + '(' \
                        + rstring_cursym_escaped + '))|' + '((' \
                        + rstring_cursym_escaped + ')\s*[\+\-]{0,1}\d+[\d\.,]*\s*(' \
                        + rstring_curmagn + '){0,1})|' + '(' \
                        + rstring_cursym_escaped \
                        + '))($|(?=[\.!?;:\-\(\)\[\]\s]+))'

            self.regex.DETECT_CURRENCY = re.compile(cur_str)

        except Exception as e:
            print('', file=sys.stderr)
            print('*** An exception occurred in section', sys._getframe().f_code.co_name,
                  'of class', type(self).__name__, '- see Traceback for details',
                  file=sys.stderr)
            print('', file=sys.stderr)
            raise e

    def _mask_acronym(self, text):
        """
        mask between each letter of an acronym with separator self.generic_config.SEP_MASK
        in order to be able to find/process those acronyms later on

        :param text: text to be scanned for acronyms
        :return: processed text
        """

        try:
            abbr_expanded = []
            for abbr in self.regex.DETECT_ABBREVIATION.finditer(text):
                if abbr.group(0) not in self.acronym_phoneme_config.EXCLUDE:
                    abbr_expanded.append(
                        (abbr.group(0), self.generic_config.SEP_MASK.join([c for c in
                            abbr.group(0).replace('.', '')
                        ])))
            for m in abbr_expanded:
                text = text.replace(m[0], m[1], 1)

            return text
        except Exception as e:
            print('', file=sys.stderr)
            print('*** An exception occurred in section', sys._getframe().f_code.co_name,
                  'of class', type(self).__name__, '- see Traceback for details',
                  file=sys.stderr)
            print('', file=sys.stderr)
            raise e

    def _acronym_phoneme_op(self, text):
        """
        TRANSLITERATE ABBREVIATONS (WITH ONLY CAPITAL LETTERS)

        :param text: text to be scanned for acronyms
        :return: processed text
        """

        try:
            abbr_expanded = []
            for abbr in self.regex.DETECT_ABBREVIATION.finditer(text):
                if abbr.group(0) not in self.acronym_phoneme_config.EXCLUDE:
                    abbr_expanded.append(
                        (abbr.group(0), self.generic_config.SEP_MASK.join([
                            self.acronym_phoneme_config.LETTER[c] for c in
                            abbr.group(0).replace('.', '')
                        ])))
            for m in abbr_expanded:
                text = text.replace(m[0], m[1], 1)

            return text
        except Exception as e:
            print('', file=sys.stderr)
            print('*** An exception occurred in section', sys._getframe().f_code.co_name,
                  'of class', type(self).__name__, '- see Traceback for details',
                  file=sys.stderr)
            print('', file=sys.stderr)
            raise e

    def _accent_peculiarity_op(self, text):
        """
        REPLACE ACCENT "PECULIARITIES" WITH ASCII counterparts

        :param text: text to be scanned for accents and peculiarities
        :return: processed text
        """

        try:
            for chars, mapped in self.generic_config.UNICODE_TO_ASCII.items():
                text = re.sub('|'.join([c for c in chars]), mapped, text)

            return text
        except Exception as e:
            print('', file=sys.stderr)
            print('*** An exception occurred in section', sys._getframe().f_code.co_name,
                  'of class', type(self).__name__, '- see Traceback for details',
                  file=sys.stderr)
            print('', file=sys.stderr)
            raise e

    def _amount_money_op(self, text):
        """
        TRANSLITERATE CURRENCIES

        :param text: text to be scanned for currency symbols
        :return: processed text
        """
        try:
            diff_len = 0
            for mc in self.regex.DETECT_CURRENCY.finditer(text):

                match_currency = self._acronym_phoneme_op(mc.group(0))
                mc_end = mc.end()

                m_symbol = self.regex.DETECT_CURRENCY_SYMBOL.search(match_currency)
                m_magnitude = self.regex.DETECT_CURRENCY_MAGNITUDE.search(match_currency)
                m_number = self.regex.DETECT_NUMBER.search(match_currency)

                number = m_number.group(0) if m_number else ''
                if not m_magnitude and ',' in number:
                    cur_symbol = self.abbreviation_config.CURRENCY_SYMBOL[
                                                        m_symbol.group(0).replace(self.generic_config.SEP_MASK, '_')]
                    number = number.replace(',', ' ' + cur_symbol + ' ')
                    dec_start = number.rfind(' ' + cur_symbol)+len(cur_symbol)+2
                    decimals = number[dec_start:]
                    # invariant: ...,00 {currency_symbol}
                    if int(decimals) == 0:
                        number = number[:dec_start]
                    # edge case: fractions have more than two digits
                    elif len(decimals) > 2:
                        number = number[:dec_start] + ' ' + decimals[0:2] + ' ' + ' '.join(decimals[2:])
                    rearranged_currency_term = number
                else:
                    rearranged_currency_term = number + ' ' if m_number else ''
                    rearranged_currency_term += m_magnitude.group(0) + ' ' if m_magnitude else ''
                    rearranged_currency_term += self.abbreviation_config.CURRENCY_SYMBOL[
                        m_symbol.group(0).replace(self.generic_config.SEP_MASK, '_')]
                text = text[:mc.start() + diff_len] + rearranged_currency_term + text[mc_end + diff_len:]
                diff_len = len(rearranged_currency_term) - (mc_end - (mc.start() + diff_len))

            return text
        except Exception as e:
            print('', file=sys.stderr)
            print('*** An exception occurred in section', sys._getframe().f_code.co_name,
                  'of class', type(self).__name__, '- see Traceback for details',
                  file=sys.stderr)
            print('', file=sys.stderr)
            raise e

    def _date_op(self, text):
        """
        TRANSLITERATE DATE (uses library num2words for transliterating numbers)

        :param text: text to be scanned for date patterns
        :return: processed text
        """

        try:
            for date_m in self.regex.DETECT_DATE.finditer(text):
                frags = date_m.group(0).split('.')
                if ' ' in frags[-1]:
                    space_split = frags[-1].strip().split(' ')
                    del (frags[-1])
                    frags.extend(space_split)
                day = num2words(frags[0], lang='de', to='ordinal')
                if date_m.start() > 1 and text[date_m.start() - 2:date_m.start()] in ('m ', 'n '):
                    day += 'n'
                if frags[1].strip() in self.abbreviation_config.MONTH:
                    month = self.abbreviation_config.MONTH[frags[1].strip()]
                else:
                    month = self.abbreviation_config.NUMBER_MONTH[frags[1].strip()]
                year = ''
                if len(frags) == 3 and frags[2]:
                    year = num2words(frags[2], lang='de', to='year')
                text = self.regex.DETECT_DATE.sub(day + ' ' + month + (' ' + year if year else ''), text, count=1)

            return text
        except Exception as e:
            print('', file=sys.stderr)
            print('*** An exception occurred in section', sys._getframe().f_code.co_name,
                  'of class', type(self).__name__, '- see Traceback for details',
                  file=sys.stderr)
            print('', file=sys.stderr)
            raise e

    def _timestamp_op(self, text):
        """
        TRANSLITERATE TIMESTAMP

        :param text:
        :return:
        """

        try:
            for timestamp_m in self.regex.DETECT_TIMESTAMP.finditer(text):
                ts = timestamp_m.group(0)
                ts_split = ts.split(':')
                if len(ts_split) == 2:
                    if int(ts_split[0].replace('h', '').replace('std', '')) == 1:
                        ts = 'eine stunde '
                    else:
                        ts = ts_split[0] + ' stunden '
                    if int(ts_split[1].replace('m', '').replace('min', '')) == 1:
                        ts += 'eine minute'
                    else:
                        ts += ts_split[1] + ' minuten'
                # assume len=3
                else:
                    if int(ts_split[0].replace('h', '').replace('std', '')) == 1:
                        ts = 'eine stunde '
                    else:
                        ts = ts_split[0] + ' stunden '
                    if int(ts_split[1].replace('min', '').replace('m', '')) == 1:
                        ts += 'eine minute '
                    else:
                        ts += ts_split[1].replace('min', '').replace('m', '') + ' minuten '
                    if int(ts_split[2].replace('sek', '').replace('sec', '').replace('s', '')) == 1:
                        ts += 'eine sekunde'
                    else:
                        ts += ts_split[2].replace('sek', '').replace('sec', '').replace('s', '') + ' sekunden'

                text = text[:timestamp_m.start()] + ts + text[timestamp_m.end():]

            return text
        except Exception as e:
            print('', file=sys.stderr)
            print('*** An exception occurred in section', sys._getframe().f_code.co_name,
                  'of class', type(self).__name__, '- see Traceback for details',
                  file=sys.stderr)
            print('', file=sys.stderr)
            raise e

    def _timeofday_op(self, text):
        """
        TRANSLITERATE TIME OF DAY

        :param text:
        :return:
        """

        try:
            # TODO: cover also other ways of transliterating time of day (e.g. 07.45h as "dreiviertel acht")
            for time_m in self.regex.DETECT_TIME_OF_DAY.finditer(text):
                tod = text[time_m.start():time_m.end()].replace('uhr', '').replace('h', '').replace(':',
                                                                                                    ' uhr ').replace(
                    '.', ' uhr ')
                if int(tod.split(' uhr ')[0]) == 1:
                    tod = 'ein uhr ' + tod.split(' uhr ')[1]
                text = text[:time_m.start()] + tod + text[time_m.end():]

            return text
        except Exception as e:
            print('', file=sys.stderr)
            print('*** An exception occurred in section', sys._getframe().f_code.co_name,
                  'of class', type(self).__name__, '- see Traceback for details',
                  file=sys.stderr)
            print('', file=sys.stderr)
            raise e

    def _weekday_op(self, word):
        """
        TRANSLITERATE WEEKDAY

        :param word:
        :return:
        """

        try:
            ms = self.regex.DETECT_WEEKDAY.finditer(word)
            offset = 0
            for m in ms:
                _word = word[m.start() + offset:m.end() + offset].replace('.', '')
                if _word in self.abbreviation_config.WEEKDAY:
                    word = word[:m.start() + offset] + self.abbreviation_config.WEEKDAY[_word] + word[m.end() + offset:]
                    offset += len(self.abbreviation_config.WEEKDAY[_word]) - (m.end() - m.start())

            if offset > 0:
                return word.replace('.', '')
            else:
                return word

        except Exception as e:
            print('', file=sys.stderr)
            print('*** An exception occurred in section', sys._getframe().f_code.co_name,
                  'of class', type(self).__name__, '- see Traceback for details',
                  file=sys.stderr)
            print('', file=sys.stderr)
            raise e

    def _month_op(self, word):
        """
        TRANSLITERATE MONTH

        :param word:
        :return: processed single word
        """

        try:
            ms = self.regex.DETECT_MONTH.finditer(word)
            offset = 0
            for m in ms:
                _word = word[m.start() + offset:m.end() + offset].replace('.', '')
                if _word in self.abbreviation_config.MONTH:
                    word = word[:m.start() + offset] + self.abbreviation_config.MONTH[_word] + word[m.end() + offset:]
                    offset += len(self.abbreviation_config.MONTH[_word]) - (m.end() - m.start())

            if offset > 0:
                return word.replace('.', '')
            else:
                return word

        except Exception as e:
            print('', file=sys.stderr)
            print('*** An exception occurred in section', sys._getframe().f_code.co_name,
                  'of class', type(self).__name__, '- see Traceback for details',
                  file=sys.stderr)
            print('', file=sys.stderr)
            raise e

    def _ordinal_op(self, word, idx, split_text, cleaned):
        """
        TRANSLITERATE ORDINAL NUMBER (uses also library num2words)

        :param word:
        :param idx: current word index in the list of words (split from full text)
        :return:
        """

        try:
            if self.regex.DETECT_ORDINAL.match(word) and word.endswith('.'):
                if idx < (len(split_text) - 1) \
                        and split_text[idx + 1] not in self.abbreviation_config.CURRENCY_SYMBOL.values():
                    word = num2words(word, lang='de', to='ordinal')
                    if idx > 0 and idx < (len(split_text) - 1):
                        if cleaned[idx - 1].endswith('m'):
                            word += 'n'

            return word

        except Exception as e:
            print('', file=sys.stderr)
            print('*** An exception occurred in section', sys._getframe().f_code.co_name,
                  'of class', type(self).__name__, '- see Traceback for details',
                  file=sys.stderr)
            print('', file=sys.stderr)
            raise e

    def _special_op(self, word):
        """
        TRANSLITERATE SPECIAL TERMS/CONSTRUCTS

        :param word:
        :return:
        """
        try:
            for pat, tup_repl in self.regex.SPECIAL_TRANSLITERATE.items():
                if re.search(pat, word):
                    word = word.replace(tup_repl[0], tup_repl[1], 1)
                    ws = []
                    for w in word.split(' '):
                        if self.regex.DETECT_NUMBER.match(w):
                            ws.append(self._transliterate_number(w))
                        else:
                            ws.append(w)
                    if ws:
                        word = ' '.join(ws)

            return word
        except Exception as e:
            print('', file=sys.stderr)
            print('*** An exception occurred in section', sys._getframe().f_code.co_name,
                  'of class', type(self).__name__, '- see Traceback for details',
                  file=sys.stderr)
            print('', file=sys.stderr)
            raise e

    def _math_symbol_op(self, word):
        """
        TRANSLITERATE SOME MATH SYMBOLS (rather experimental)

        :param word: single word to be scanned for math symbols
        :return: processed single word
        """

        try:
            for pat, repl in self.abbreviation_config.MATH_SYMBOL.items():
                if pat in word:
                    # we need heuristics... TODO: having better idea?
                    if pat == 'x':
                        if len(word) > 1 and word.find(pat) == 0:
                            continue
                        elif len(word) > 1 and not word[:word.find(pat)].isdecimal():
                            continue
                    elif pat == '-':
                        if word == '--':
                            continue
                        elif len(word) > 1 and \
                                (len(word[:word.find(pat)]) > 1 or len(word[word.find(pat) + 1:]) > 1):
                            continue

                    word = word.replace(pat, repl, 1)

            return word
        except Exception as e:
            print('', file=sys.stderr)
            print('*** An exception occurred in section', sys._getframe().f_code.co_name,
                  'of class', type(self).__name__, '- see Traceback for details',
                  file=sys.stderr)
            print('', file=sys.stderr)
            raise e

    def _spoken_symbol_op(self, word, idx, split_text):
        """
        TRANSLITERATE SPOKEN SYMBOLS (e.g. transliterate brackets into "in Klammern"),
        mainly useful for TTS tasks (or ASR preprocessing for trainings)

        :param word: single word to be scanned for spoken symbols
        :param idx: current word index in the list of words (split from full text)
        :param split_text: list of words (where parameter word is part of)
        :return: processed word
        """

        try:
            for pats, repl in self.abbreviation_config.SPOKEN_SYMBOL.items():
                if pats[0] in word:
                    word = word.replace(pats[0], repl.replace('_', self.generic_config.SEP_MASK))
                    if pats[1] in word:
                        word = word.replace(pats[1], self.generic_config.SEP_MASK)
                    else:
                        for fwd_idx in range(idx + 1, len(split_text)):
                            if pats[1] in split_text[fwd_idx]:
                                split_text[fwd_idx] = split_text[fwd_idx].replace(pats[1],
                                                                                  self.generic_config.SEP_MASK)
                                break

            return word
        except Exception as e:
            print('', file=sys.stderr)
            print('*** An exception occurred in section', sys._getframe().f_code.co_name,
                  'of class', type(self).__name__, '- see Traceback for details',
                  file=sys.stderr)
            print('', file=sys.stderr)
            raise e

    def _misc_abbreviation_op(self, word):
        """
        TRANSLITERATE MISCELLANEOUS TERMS

        :param word: single word to be scanned for various terms or symbols
        :return: processed single word
        """

        try:
            for short, long in self.abbreviation_config.MISC.items():
                if long not in word:
                    # invariant: cover also abbreviation with '.' at the end
                    if (short + '.') == word or short == word:
                        word = long

            return word
        except Exception as e:
            print('', file=sys.stderr)
            print('*** An exception occurred in section', sys._getframe().f_code.co_name,
                  'of class', type(self).__name__, '- see Traceback for details',
                  file=sys.stderr)
            print('', file=sys.stderr)
            raise e

    def _number_unit_op(self, word, idx, cleaned_words):
        """
        TRANSLITERATE MISC: UNITS in conjunction with numbers, either units attached to the number or separated;
        takes into account if singular or plural is required

        :param word: a single word to be transliterated
        :param idx:  current word index in the list of words (split from full text)
        :param cleaned_words: list of words already processed before
        :return: processed word (may be more than one actually due to number-unit expansion!)
        """

        # EXPAND UNITS *SEPARATED* from numbers (as opposed to *ATTACHED* to numbers, see below)
        # e.g. "t" in conjunction with term "20 t"

        try:
            w_unit = word
            # TODO: remove non-alphanum for mapping to transliterations, e.g. boundary chars like "!" or "?"
            if w_unit.endswith('.'):
                w_unit = w_unit[:-1]
            w_unit = w_unit.replace(self.generic_config.SEP_MASK, '_')

            if w_unit in self.unit_config.PLURAL_NO_SUFFIX.keys():
                word = self.unit_config.PLURAL_NO_SUFFIX[w_unit]
                if idx > 0 and cleaned_words[idx - 1] == ('eins'):
                    cleaned_words[idx - 1] = 'ein'
            elif w_unit in self.unit_config.PLURAL_SUFFIX_EN.keys():
                word = self.unit_config.PLURAL_SUFFIX_EN[w_unit]
                # invariant: one <unit>
                if idx > 0 and (cleaned_words[idx - 1] == ('eins')
                                or cleaned_words[idx - 1] == ('ein')
                                or cleaned_words[idx - 1] == ('eine')
                                or cleaned_words[idx - 1] == ('einer')
                                or cleaned_words[idx - 1] == ('einen')
                                or cleaned_words[idx - 1] == ('einem')):
                    word = word[:-2]
                    cleaned_words[idx - 1] = 'eine'

            elif w_unit in self.unit_config.PLURAL_SUFFIX_N.keys():
                word = self.unit_config.PLURAL_SUFFIX_N[w_unit]
                # invariant: one <unit>
                if idx > 0 and (cleaned_words[idx - 1] == ('eins')
                                or cleaned_words[idx - 1] == ('ein')
                                or cleaned_words[idx - 1] == ('eine')
                                or cleaned_words[idx - 1] == ('einer')
                                or cleaned_words[idx - 1] == ('einen')
                                or cleaned_words[idx - 1] == ('einem')):
                    word = word[:-1]
                    cleaned_words[idx - 1] = 'eine'

            # EXPAND (remaining) NUMBERS
            num_match = self.regex.DETECT_NUMBER.match(word)
            if num_match:

                # EXPAND UNITS *ATTACHED* to numbers (as opposed to *SEPARATED* from numbers, see above)
                # e.g. "20t"

                w_unit = word[num_match.end():]
                invariant_one = False
                if w_unit:
                    if w_unit.endswith('.'):
                        w_unit = w_unit[:-1]

                    if w_unit in self.unit_config.PLURAL_NO_SUFFIX.keys():
                        w_unit = self.unit_config.PLURAL_NO_SUFFIX[w_unit]
                    elif w_unit in self.unit_config.PLURAL_SUFFIX_EN.keys():
                        w_unit = self.unit_config.PLURAL_SUFFIX_EN[w_unit]
                        # invariant: one <unit>
                        if num_match.group(0).replace('+', '').replace('-', '') in ['1', '1.0', '1.00']:
                            w_unit = w_unit[:-2]
                            invariant_one = True
                    elif w_unit in self.unit_config.PLURAL_SUFFIX_N.keys():
                        w_unit = self.unit_config.PLURAL_SUFFIX_N[w_unit]
                        # invariant: one <unit>
                        if num_match.group(0).replace('+', '').replace('-', '') in ['1', '1.0', '1.00']:
                            w_unit = w_unit[:-1]
                            invariant_one = True
                    w_unit = ' ' + w_unit
                    word = num_match.group(0)

                if invariant_one:
                    word = 'eine'
                else:
                    word = self._transliterate_number(word)
                word += w_unit

            return word
        except Exception as e:
            print('', file=sys.stderr)
            print('*** An exception occurred in section', sys._getframe().f_code.co_name,
                  'of class', type(self).__name__, '- see Traceback for details',
                  file=sys.stderr)
            print('', file=sys.stderr)
            raise e

    def transliterate(self, text):
        """
        Transliterates any text to a normalized form as given by config (cf. parameter transliterate_ops in constructor)

        :param text: the (messy, not normalized, ...) text to be transliterated to a normalized form
        :return: the normalized form of the given text
        """

        # GENERAL NOTE: The order of transliterate_ops executed (set via input config parameter transliterate_ops)
        #               is IMPORTANT and not interchangeable!

        # First: do replacing and normalizing (transliterating) on the full text

        try:

            text = self._mask_acronym(text)
            if 'acronym_phoneme' in self.transliterate_ops:
                text = self._acronym_phoneme_op(text)

            # MAKE LOWERCASE (only AFTER acronym processing!)
            if self.make_lowercase:
                text = text.lower()

            if 'accent_peculiarity' in self.transliterate_ops:
                text = self._accent_peculiarity_op(text)

            if 'amount_money' in self.transliterate_ops:
                text = self._amount_money_op(text)

            # TODO (OPTIONAL): cover also time durations
            if 'date' in self.transliterate_ops:
                text = self._date_op(text)

            if 'timestamp' in self.transliterate_ops:
                text = self._timestamp_op(text)

            if 'time_of_day' in self.transliterate_ops:
                text = self._timeofday_op(text)

            # Second: iterate over all single words of the text (split by space character)

            # split full text (as list)
            split_text = text.split(' ')
            # list of cleaned/normalized/transliterated words (as derived from split_text)
            cleaned_words = []

            idx = 0
            for word in split_text:
                if not word:
                    continue

                # EXPAND CHARACTERS AND TERMS for transliteration
                for tr in self.transliterate_ops:
                    # WEEKDAYS
                    if tr == 'weekday':
                        word = self._weekday_op(word)
                    # MONTH
                    elif tr == 'month':
                        word = self._month_op(word)

                    # ORDINAL NUMBERS
                    elif tr == 'ordinal':
                        word = self._ordinal_op(word, idx, split_text, cleaned_words)

                    # SPECIAL TERMS or CHARACTERS
                    # TODO: because of the kind of split, only statements/terms with a single word
                    #       are covered currently (e.g. "8/10" but not "8 / 10")
                    elif tr == 'special':
                        word = self._special_op(word)

                    # SOME MATH SYMBOLS
                    elif tr == 'math_symbol':
                        word = self._math_symbol_op(word)

                    # SPOKEN SYMBOLS (e.g. brackets as "in Klammern")
                    elif tr == 'spoken_symbol':
                        word = self._spoken_symbol_op(word, idx, split_text)

                # REPLACE/MAP (remaining) SPECIFIC TERMS/CHARACTERS
                for old, new in self.replace.items():
                    word = word.replace(old, new)

                # REPLACE/MAP miscellaneous abbreviations or short forms
                word = self._misc_abbreviation_op(word)

                # TRANSLITERATE NUMBERS in combination with misc. units (both attached and separated units)
                word = self._number_unit_op(word, idx, cleaned_words)

                cleaned_words.append(word)
                idx += 1

            # WRAP UP

            # 1) merge list back to single string; also replace remaining masked separators with one according to config parameter
            text = ' '.join(cleaned_words).replace(self.generic_config.SEP_MASK, self.sep_abbreviation)
            # 2) remove multiple whitespaces (i.e. replace with just one whitespace)
            return self.regex.DETECT_WHITESPACE_SEQ.sub(' ', text)
        except Exception as e:
            print('', file=sys.stderr)
            print('*** An exception occurred in section', sys._getframe().f_code.co_name,
                  'of class', type(self).__name__, '- see Traceback for details',
                  file=sys.stderr)
            print('', file=sys.stderr)
            raise e

    def _transliterate_number(self, number: str) -> str:
        """
        Uses library num2words for transliterating numbers (also floats or with decimal point numbers)

        IMPORTANT NOTE: GERMAN version means 1000's marks is "."
        and decimal point is "," (the opposite of in English)

        :param number: the number word to be transliterated
        :return: transliterated given number
        """

        try:
            # invariant: 1000's mark(s) AND floating point number
            if number.count(',') == 1 and number.count('.') >= 1:
                number = number.replace('.', '')

            try:
                # floating number only
                if number.count(',') == 1:
                    number = number.replace(',', '.')
                    word = num2words(float(number), lang='de', to='cardinal').lower()
                # 1000's marks only
                elif number.count('.') >= 1:
                    number = number.replace('.', '')
                    word = num2words(int(number), lang='de', to='cardinal').lower()
                # integer only
                else:
                    word = num2words(int(number), lang='de', to='cardinal').lower()
            except ValueError:
                # ignore HERE: mixed numbers are handled further down in the pipeline!
                word = number
            return word
        except Exception as e:
            print('', file=sys.stderr)
            print('*** An exception occurred in section', sys._getframe().f_code.co_name,
                  'of class', type(self).__name__, '- see Traceback for details',
                  file=sys.stderr)
            print('', file=sys.stderr)
            raise e


_inflect = inflect.engine()
_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')


def _remove_commas(m):
  return m.group(1).replace(',', '')


def _expand_decimal_point(m):
  return m.group(1).replace('.', ' point ')


def _expand_dollars(m):
  match = m.group(1)
  parts = match.split('.')
  if len(parts) > 2:
    return match + ' dollars'  # Unexpected format
  dollars = int(parts[0]) if parts[0] else 0
  cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
  if dollars and cents:
    dollar_unit = 'dollar' if dollars == 1 else 'dollars'
    cent_unit = 'cent' if cents == 1 else 'cents'
    return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
  elif dollars:
    dollar_unit = 'dollar' if dollars == 1 else 'dollars'
    return '%s %s' % (dollars, dollar_unit)
  elif cents:
    cent_unit = 'cent' if cents == 1 else 'cents'
    return '%s %s' % (cents, cent_unit)
  else:
    return 'zero dollars'


def _expand_ordinal(m):
  return _inflect.number_to_words(m.group(0))


def _expand_number(m):
  num = int(m.group(0))
  if num > 1000 and num < 3000:
    if num == 2000:
      return 'two thousand'
    elif num > 2000 and num < 2010:
      return 'two thousand ' + _inflect.number_to_words(num % 100)
    elif num % 100 == 0:
      return _inflect.number_to_words(num // 100) + ' hundred'
    else:
      return _inflect.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
  else:
    return _inflect.number_to_words(num, andword='')


def normalize_numbers(text):
  text = re.sub(_comma_number_re, _remove_commas, text)
  text = re.sub(_pounds_re, r'\1 pounds', text)
  text = re.sub(_dollars_re, _expand_dollars, text)
  text = re.sub(_decimal_number_re, _expand_decimal_point, text)
  text = re.sub(_ordinal_re, _expand_ordinal, text)
  text = re.sub(_number_re, _expand_number, text)
  return text


def expand_numbers(text):
  return normalize_numbers(text)

def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text


def lowercase(text):
  return text.lower()


def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
  return unidecode(text)



def japanese_to_romaji_with_accent(text):
  '''Reference https://r9y9.github.io/ttslearn/latest/notebooks/ch10_Recipe-Tacotron.html'''
  sentences = re.split(_japanese_marks, text)
  marks = re.findall(_japanese_marks, text)
  text = ''
  for i, sentence in enumerate(sentences):
    if re.match(_japanese_characters, sentence):
      if text!='':
        text+=' '
      labels = pyopenjtalk.extract_fullcontext(sentence)
      for n, label in enumerate(labels):
        phoneme = re.search(r'\-([^\+]*)\+', label).group(1)
        if phoneme not in ['sil','pau']:
          text += phoneme.replace('ch','ʧ').replace('sh','ʃ').replace('cl','Q')
        else:
          continue
        n_moras = int(re.search(r'/F:(\d+)_', label).group(1))
        a1 = int(re.search(r"/A:(\-?[0-9]+)\+", label).group(1))
        a2 = int(re.search(r"\+(\d+)\+", label).group(1))
        a3 = int(re.search(r"\+(\d+)/", label).group(1))
        if re.search(r'\-([^\+]*)\+', labels[n + 1]).group(1) in ['sil','pau']:
          a2_next=-1
        else:
          a2_next = int(re.search(r"\+(\d+)\+", labels[n + 1]).group(1))
        # Accent phrase boundary
        if a3 == 1 and a2_next == 1:
          text += ' '
        # Falling
        elif a1 == 0 and a2_next == a2 + 1 and a2 != n_moras:
          text += '↓'
        # Rising
        elif a2 == 1 and a2_next == 2:
          text += '↑'
    if i<len(marks):
      text += unidecode(marks[i]).replace(' ','')
  return text


def latin_to_hangul(text):
  for regex, replacement in _latin_to_hangul:
    text = re.sub(regex, replacement, text)
  return text


def divide_hangul(text):
  for regex, replacement in _hangul_divided:
    text = re.sub(regex, replacement, text)
  return text


def hangul_number(num, sino=True):
  '''Reference https://github.com/Kyubyong/g2pK'''
  num = re.sub(',', '', num)

  if num == '0':
      return '영'
  if not sino and num == '20':
      return '스무'

  digits = '123456789'
  names = '일이삼사오육칠팔구'
  digit2name = {d: n for d, n in zip(digits, names)}
  
  modifiers = '한 두 세 네 다섯 여섯 일곱 여덟 아홉'
  decimals = '열 스물 서른 마흔 쉰 예순 일흔 여든 아흔'
  digit2mod = {d: mod for d, mod in zip(digits, modifiers.split())}
  digit2dec = {d: dec for d, dec in zip(digits, decimals.split())}

  spelledout = []
  for i, digit in enumerate(num):
    i = len(num) - i - 1
    if sino:
      if i == 0:
        name = digit2name.get(digit, '')
      elif i == 1:
        name = digit2name.get(digit, '') + '십'
        name = name.replace('일십', '십')
    else:
      if i == 0:
        name = digit2mod.get(digit, '')
      elif i == 1:
        name = digit2dec.get(digit, '')
    if digit == '0':
      if i % 4 == 0:
        last_three = spelledout[-min(3, len(spelledout)):]
        if ''.join(last_three) == '':
          spelledout.append('')
          continue
      else:
        spelledout.append('')
        continue
    if i == 2:
      name = digit2name.get(digit, '') + '백'
      name = name.replace('일백', '백')
    elif i == 3:
      name = digit2name.get(digit, '') + '천'
      name = name.replace('일천', '천')
    elif i == 4:
      name = digit2name.get(digit, '') + '만'
      name = name.replace('일만', '만')
    elif i == 5:
      name = digit2name.get(digit, '') + '십'
      name = name.replace('일십', '십')
    elif i == 6:
      name = digit2name.get(digit, '') + '백'
      name = name.replace('일백', '백')
    elif i == 7:
      name = digit2name.get(digit, '') + '천'
      name = name.replace('일천', '천')
    elif i == 8:
      name = digit2name.get(digit, '') + '억'
    elif i == 9:
      name = digit2name.get(digit, '') + '십'
    elif i == 10:
      name = digit2name.get(digit, '') + '백'
    elif i == 11:
      name = digit2name.get(digit, '') + '천'
    elif i == 12:
      name = digit2name.get(digit, '') + '조'
    elif i == 13:
      name = digit2name.get(digit, '') + '십'
    elif i == 14:
      name = digit2name.get(digit, '') + '백'
    elif i == 15:
      name = digit2name.get(digit, '') + '천'
    spelledout.append(name)
  return ''.join(elem for elem in spelledout)


def number_to_hangul(text):
  '''Reference https://github.com/Kyubyong/g2pK'''
  tokens = set(re.findall(r'(\d[\d,]*)([\uac00-\ud71f]+)', text))
  for token in tokens:
    num, classifier = token
    if classifier[:2] in _korean_classifiers or classifier[0] in _korean_classifiers:
      spelledout = hangul_number(num, sino=False)
    else:
      spelledout = hangul_number(num, sino=True)
    text = text.replace(f'{num}{classifier}', f'{spelledout}{classifier}')
  # digit by digit for remaining digits
  digits = '0123456789'
  names = '영일이삼사오육칠팔구'
  for d, n in zip(digits, names):
    text = text.replace(d, n)
  return text


def number_to_chinese(text):
  numbers = re.findall(r'\d+(?:\.?\d+)?', text)
  for number in numbers:
    text = text.replace(number, cn2an.an2cn(number),1)
  return text


def chinese_to_bopomofo(text):
  text=text.replace('、','，').replace('；','，').replace('：','，')
  words=jieba.lcut(text,cut_all=False)
  text=''
  for word in words:
    bopomofos=lazy_pinyin(word,BOPOMOFO)
    if not re.search('[\u4e00-\u9fff]',word):
      text+=word
      continue
    for i in range(len(bopomofos)):
      if re.match('[\u3105-\u3129]',bopomofos[i][-1]):
        bopomofos[i]+='ˉ'
    if text!='':
      text+=' '
    text+=''.join(bopomofos)
  return text


def latin_to_bopomofo(text):
  for regex, replacement in _latin_to_bopomofo:
    text = re.sub(regex, replacement, text)
  return text


def bopomofo_to_romaji(text):
  for regex, replacement in _bopomofo_to_romaji:
    text = re.sub(regex, replacement, text)
  return text


def basic_cleaners(text):
  '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def transliteration_cleaners(text):
  '''Pipeline for non-English text that transliterates to ASCII.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def japanese_cleaners(text):
  text=japanese_to_romaji_with_accent(text)
  if re.match('[A-Za-z]',text[-1]):
    text += '.'
  return text


def japanese_cleaners2(text):
  return japanese_cleaners(text).replace('ts','ʦ')


def korean_cleaners(text):
  '''Pipeline for Korean text'''
  text = latin_to_hangul(text)
  text = number_to_hangul(text)
  text = j2hcj(h2j(text))
  text = divide_hangul(text)
  if re.match('[\u3131-\u3163]',text[-1]):
    text += '.'
  return text


def chinese_cleaners(text):
  '''Pipeline for Chinese text'''
  text=number_to_chinese(text)
  text=chinese_to_bopomofo(text)
  text=latin_to_bopomofo(text)
  if re.match('[ˉˊˇˋ˙]',text[-1]):
    text += '。'
  return text


def zh_ja_mixture_cleaners(text):
  chinese_texts=re.findall(r'\[ZH\].*?\[ZH\]',text)
  japanese_texts=re.findall(r'\[JA\].*?\[JA\]',text)
  for chinese_text in chinese_texts:
    cleaned_text=number_to_chinese(chinese_text[4:-4])
    cleaned_text=chinese_to_bopomofo(cleaned_text)
    cleaned_text=latin_to_bopomofo(cleaned_text)
    cleaned_text=bopomofo_to_romaji(cleaned_text)
    cleaned_text=re.sub('i[aoe]',lambda x:'y'+x.group(0)[1:],cleaned_text)
    cleaned_text=re.sub('u[aoəe]',lambda x:'w'+x.group(0)[1:],cleaned_text)
    cleaned_text=re.sub('([ʦsɹ]`[⁼ʰ]?)([→↓↑]+)',lambda x:x.group(1)+'ɹ`'+x.group(2),cleaned_text).replace('ɻ','ɹ`')
    cleaned_text=re.sub('([ʦs][⁼ʰ]?)([→↓↑]+)',lambda x:x.group(1)+'ɹ'+x.group(2),cleaned_text)
    text = text.replace(chinese_text,cleaned_text+' ',1)
  for japanese_text in japanese_texts:
    cleaned_text=japanese_to_romaji_with_accent(japanese_text[4:-4]).replace('ts','ʦ').replace('u','ɯ').replace('...','…')
    text = text.replace(japanese_text,cleaned_text+' ',1)
  text=text[:-1]
  if re.match('[A-Za-zɯɹəɥ→↓↑]',text[-1]):
    text += '.'
  return text

def english_cleaners(text):
  '''Pipeline for English text, including abbreviation expansion.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_abbreviations(text)
  phonemes = phonemize(text, language='en-us', backend='espeak', strip=True)
  phonemes = collapse_whitespace(phonemes)
  return phonemes

def english_cleaners2(text):
  '''Pipeline for English text, including abbreviation expansion. + punctuation + stress'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_abbreviations(text)
  phonemes = phonemize(text, language='en-us', backend='espeak', strip=True, preserve_punctuation=True, with_stress=True)
  phonemes = collapse_whitespace(phonemes)
  return phonemes

def german_cleaners(text):
   ops = {'accent_peculiarity', 'amount_money', 'date', 'timestamp', 'time_of_day', 'ordinal', 'special'}
   normalized_text = GermanTransliterate(transliterate_ops=ops).transliterate(text)
   return normalized_text


def zh_ja_en_de_mixture_cleaners(text):
  chinese_texts=re.findall(r'\[ZH\].*?\[ZH\]',text)
  japanese_texts=re.findall(r'\[JA\].*?\[JA\]',text)
  english_texts=re.findall(r'\[EN\].*?\[EN\]',text)
  german_texts=re.findall(r'\[DE\].*?\[DE\]',text)
  for chinese_text in chinese_texts:
    cleaned_text=number_to_chinese(chinese_text[4:-4])
    cleaned_text=chinese_to_bopomofo(cleaned_text)
    cleaned_text=latin_to_bopomofo(cleaned_text)
    cleaned_text=bopomofo_to_romaji(cleaned_text)
    cleaned_text=re.sub('i[aoe]',lambda x:'y'+x.group(0)[1:],cleaned_text)
    cleaned_text=re.sub('u[aoəe]',lambda x:'w'+x.group(0)[1:],cleaned_text)
    cleaned_text=re.sub('([ʦsɹ]`[⁼ʰ]?)([→↓↑]+)',lambda x:x.group(1)+'ɹ`'+x.group(2),cleaned_text).replace('ɻ','ɹ`')
    cleaned_text=re.sub('([ʦs][⁼ʰ]?)([→↓↑]+)',lambda x:x.group(1)+'ɹ'+x.group(2),cleaned_text)
    text = text.replace(chinese_text,cleaned_text+' ',1)
  for japanese_text in japanese_texts:
    cleaned_text=japanese_to_romaji_with_accent(japanese_text[4:-4]).replace('ts','ʦ').replace('u','ɯ').replace('...','…')
    text = text.replace(japanese_text,cleaned_text+' ',1)
  for english_text in english_texts:
    cleaned_text = convert_to_ascii(english_text)
    cleaned_text = lowercase(english_text)
    cleaned_text = expand_numbers(english_text)
    cleaned_text = expand_abbreviations(english_text)
    cleaned_text = phonemize(cleaned_text, language='en-us', backend='espeak', strip=True)
    cleaned_text = collapse_whitespace(cleaned_text)
    text = text.replace(english_text,cleaned_text+' ',1)
  for german_text in german_texts:
    ops = {'accent_peculiarity', 'amount_money', 'date', 'timestamp', 'time_of_day', 'ordinal', 'special'}
    cleaned_text= GermanTransliterate(transliterate_ops=ops).transliterate(german_text)
    cleaned_text = phonemize(cleaned_text, language='de', backend='espeak', strip=True)
    text = text.replace(german_text,cleaned_text+' ',1)
  text=text[:-1]
  if re.match('[A-Za-zɯɹəɥ→↓↑]',text[-1]):
    text += '.'
  return text
