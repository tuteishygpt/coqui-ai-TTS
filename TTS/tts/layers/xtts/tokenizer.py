# V1 (with detailed logging)
import logging
import os
import re
import textwrap
import inspect
from functools import cached_property

import torch
from ko_speech_tools import hangul_romanize
from num2words import num2words
from tokenizers import Tokenizer

from TTS.tts.layers.xtts.zh_num2words import TextNorm as zh_num2words
from TTS.tts.utils.text.cleaners import collapse_whitespace, lowercase

logger = logging.getLogger(__name__)


def _fn():
    """Helper to print fully-qualified function name in logs."""
    frame = inspect.currentframe().f_back
    mod = frame.f_globals.get("__name__", "")
    return f"{mod}.{frame.f_code.co_name}"


def _log_chunks(chunks, origin):
    """Log each chunk's index, length, and text."""
    logger.info("%s: produced %d chunk(s)", origin, len(chunks))
    for i, c in enumerate(chunks, 1):
        logger.info("%s: chunk %d -> %d chars | %s", origin, i, len(c), c)


def get_spacy_lang(lang):
    try:
        from spacy.lang.ar import Arabic
        from spacy.lang.en import English
        from spacy.lang.es import Spanish
        from spacy.lang.hi import Hindi
        from spacy.lang.ja import Japanese
        from spacy.lang.zh import Chinese
    except ImportError as e:
        raise ImportError("enable_text_splitting=True requires Spacy: pip install spacy[ja]") from e
    """Return Spacy language used for sentence splitting."""
    if lang == "zh":
        return Chinese()
    elif lang == "ja":
        return Japanese()
    elif lang == "ar":
        return Arabic()
    elif lang == "es":
        return Spanish()
    elif lang == "hi":
        return Hindi()
    else:
        # For most languages, English does the job
        return English()


def split_sentence(text, lang, text_split_length=250, min_chunk_length=40):
    """Split text into chunks using sentence boundaries and length limits.

    Дадаткова: стараемся не ствараць кавалкі карацей за min_chunk_length сімвалаў
    (калі кавалкаў больш за адзін). Калі для гэтага трэба, апошні кавалак
    можа перавысіць text_split_length.
    """
    origin = _fn()
    text = text.strip()
    logger.info("%s: called with lang=%s, text_len=%d, text_split_length=%s, min_chunk_length=%s",
                origin, lang, len(text), text_split_length, min_chunk_length)

    if not text:
        logger.info("%s: empty text -> []", origin)
        return []

    # Калі ліміт не заданы або тэкст і так карацейшы — вяртаем як ёсць
    if text_split_length is None or len(text) <= text_split_length:
        logger.info("%s: early-return (len(text) <= text_split_length)", origin)
        chunks = [text]
        _log_chunks(chunks, origin)
        return chunks

    # 1. Спачатку рэжам па сказах і ліміце даўжыні
    nlp = get_spacy_lang(lang)
    nlp.add_pipe("sentencizer")
    logger.info("%s: using spaCy sentencizer (%s)", origin, type(nlp).__name__)

    doc = nlp(text)

    raw_chunks = []
    current = ""

    for sentence in doc.sents:
        sentence = str(sentence).strip()
        if not sentence:
            continue

        # Калі адзін сказ даўжэй за ліміт — рэжам textwrap’ам
        if len(sentence) > text_split_length:
            # спачатку захоўваем тое, што ўжо назбіралася
            if current:
                raw_chunks.append(current)
                current = ""
            logger.info("%s: long sentence (%d) > %d -> textwrap.wrap", origin, len(sentence), text_split_length)
            for line in textwrap.wrap(
                sentence,
                width=text_split_length,
                drop_whitespace=True,
                break_on_hyphens=False,
                tabsize=1,
            ):
                line = line.strip()
                if line:
                    raw_chunks.append(line)
            continue

        # Стараемся прыляпіць сказ да бягучага кавалка
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= text_split_length:
            current = (current + " " + sentence).strip()
        else:
            raw_chunks.append(current)
            current = sentence

    if current:
        raw_chunks.append(current)

    logger.info("%s: raw_chunks after sentence pass -> %d", origin, len(raw_chunks))

    # 2. Падчэсваем кавалкі: не пакідаем кавалкаў < min_chunk_length,
    # калі толькі гэта не адзіны кавалак.
    if len(raw_chunks) <= 1:
        chunks = [c.lstrip() for c in raw_chunks]
        _log_chunks(chunks, origin)
        return chunks

    merged_chunks = []
    i = 0
    while i < len(raw_chunks):
        chunk = raw_chunks[i]
        # Пакуль кавалак карацейшы за min_chunk_length і ёсць наступныя — зліваём
        while len(chunk) < min_chunk_length and i + 1 < len(raw_chunks):
            logger.info("%s: merging short chunk (%d < %d) with next", origin, len(chunk), min_chunk_length)
            i += 1
            chunk = (chunk + " " + raw_chunks[i]).strip()
        merged_chunks.append(chunk)
        i += 1

    # Калі апошні кавалак усё яшчэ карацейшы за мінімум і кавалкаў > 1 —
    # прылепім яго да папярэдняга (ён можа перайсці ліміт text_split_length).
    if len(merged_chunks) > 1 and len(merged_chunks[-1]) < min_chunk_length:
        logger.info("%s: last chunk still short (%d < %d) -> append to previous (may exceed limit)",
                    origin, len(merged_chunks[-1]), min_chunk_length)
        merged_chunks[-2] = (merged_chunks[-2] + " " + merged_chunks[-1]).strip()
        merged_chunks.pop()

    chunks = [c.lstrip() for c in merged_chunks]
    _log_chunks(chunks, origin)
    return chunks


# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = {
    "en": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            ("mrs", "misess"),
            ("mr", "mister"),
            ("dr", "doctor"),
            ("st", "saint"),
            ("co", "company"),
            ("jr", "junior"),
            ("maj", "major"),
            ("gen", "general"),
            ("drs", "doctors"),
            ("rev", "reverend"),
            ("lt", "lieutenant"),
            ("hon", "honorable"),
            ("sgt", "sergeant"),
            ("capt", "captain"),
            ("esq", "esquire"),
            ("ltd", "limited"),
            ("col", "colonel"),
            ("ft", "fort"),
        ]
    ],
    "es": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            ("sra", "señora"),
            ("sr", "señor"),
            ("dr", "doctor"),
            ("dra", "doctora"),
            ("st", "santo"),
            ("co", "compañía"),
            ("jr", "junior"),
            ("ltd", "limitada"),
        ]
    ],
    "fr": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            ("mme", "madame"),
            ("mr", "monsieur"),
            ("dr", "docteur"),
            ("st", "saint"),
            ("co", "compagnie"),
            ("jr", "junior"),
            ("ltd", "limitée"),
        ]
    ],
    "de": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            ("fr", "frau"),
            ("dr", "doktor"),
            ("st", "sankt"),
            ("co", "firma"),
            ("jr", "junior"),
        ]
    ],
    "pt": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            ("sra", "senhora"),
            ("sr", "senhor"),
            ("dr", "doutor"),
            ("dra", "doutora"),
            ("st", "santo"),
            ("co", "companhia"),
            ("jr", "júnior"),
            ("ltd", "limitada"),
        ]
    ],
    "it": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            # ("sig.ra", "signora"),
            ("sig", "signore"),
            ("dr", "dottore"),
            ("st", "santo"),
            ("co", "compagnia"),
            ("jr", "junior"),
            ("ltd", "limitata"),
        ]
    ],
    "pl": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            ("p", "pani"),
            ("m", "pan"),
            ("dr", "doktor"),
            ("sw", "święty"),
            ("jr", "junior"),
        ]
    ],
    "ar": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            # There are not many common abbreviations in Arabic as in English.
        ]
    ],
    "zh": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            # Chinese doesn't typically use abbreviations in the same way as Latin-based scripts.
        ]
    ],
    "cs": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            ("dr", "doktor"),
            ("ing", "inženýr"),
            ("p", "pan"),
        ]
    ],
    "ru": [
        (re.compile(f"\\b{x[0]}\\b", re.IGNORECASE), x[1])
        for x in [
            ("г-жа", "госпожа"),
            ("г-н", "господин"),
            ("д-р", "доктор"),
        ]
    ],
    "nl": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            ("dhr", "de heer"),
            ("mevr", "mevrouw"),
            ("dr", "dokter"),
            ("jhr", "jonkheer"),
        ]
    ],
    "tr": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            ("b", "bay"),
            ("byk", "büyük"),
            ("dr", "doktor"),
        ]
    ],
    "hu": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            ("dr", "doktor"),
            ("b", "bácsi"),
            ("nőv", "nővér"),
        ]
    ],
    "ko": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            # Korean doesn't typically use abbreviations in the same way as Latin-based scripts.
        ]
    ],
    "hi": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            # Hindi doesn't typically use abbreviations in the same way as Latin-based scripts.
        ]
    ],
    "be": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            ("г-ня", "гаспадыня"),
            ("г-р", "гаспадар"),
            ("д-р", "доктар"),
            ("праф.", "прафесар"),
            ("гл.", "глядзі"),
            ("сп.", "спадар"),
            ("сп-ня", "спадарыня"),
            ("ст.", "старонка"),
            ("цв.", "цвёрды"),
            ("лг.", "лагер"),
            ("т-ва", "таварыства"),
            ("н-д", "напрыклад"),
            ("арт.", "артыкул"),
            ("б-р", "бульвар"),
            ("гр.", "грамадзянін"),
            ("інж.", "інжынер"),
            ("ак.", "акадэмік"),
            ("пп.", "пункт"),
            ("г.", "горад"),
            ("вобл.", "вобласць"),
            ("р-н", "раён"),
            ("м-р", "магістр"),
        ]
    ],
}


def expand_abbreviations_multilingual(text, lang="en"):
    for regex, replacement in _abbreviations[lang]:
        text = re.sub(regex, replacement, text)
    return text


_symbols_multilingual = {
    "en": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " and "),
            ("@", " at "),
            ("%", " percent "),
            ("#", " hash "),
            ("$", " dollar "),
            ("£", " pound "),
            ("°", " degree "),
        ]
    ],
    "es": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " y "),
            ("@", " arroba "),
            ("%", " por ciento "),
            ("#", " numeral "),
            ("$", " dolar "),
            ("£", " libra "),
            ("°", " grados "),
        ]
    ],
    "fr": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " et "),
            ("@", " arobase "),
            ("%", " pour cent "),
            ("#", " dièse "),
            ("$", " dollar "),
            ("£", " livre "),
            ("°", " degrés "),
        ]
    ],
    "de": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " und "),
            ("@", " at "),
            ("%", " prozent "),
            ("#", " raute "),
            ("$", " dollar "),
            ("£", " pfund "),
            ("°", " grad "),
        ]
    ],
    "pt": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " e "),
            ("@", " arroba "),
            ("%", " por cento "),
            ("#", " cardinal "),
            ("$", " dólar "),
            ("£", " libra "),
            ("°", " graus "),
        ]
    ],
    "it": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " e "),
            ("@", " chiocciola "),
            ("%", " per cento "),
            ("#", " cancelletto "),
            ("$", " dollaro "),
            ("£", " sterlina "),
            ("°", " grади "),
        ]
    ],
    "pl": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " i "),
            ("@", " małpa "),
            ("%", " procent "),
            ("#", " krzyżyk "),
            ("$", " dolar "),
            ("£", " funt "),
            ("°", " stopnie "),
        ]
    ],
    "ar": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " و "),
            ("@", " على "),
            ("%", " في المئة "),
            ("#", " رقم "),
            ("$", " دولار "),
            ("£", " جنيه "),
            ("°", " درجة "),
        ]
    ],
    "zh": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " 和 "),
            ("@", " 在 "),
            ("%", " 百分之 "),
            ("#", " 号 "),
            ("$", " 美元 "),
            ("£", " 英镑 "),
            ("°", " 度 "),
        ]
    ],
    "cs": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " a "),
            ("@", " na "),
            ("%", " procento "),
            ("#", " křížek "),
            ("$", " dolar "),
            ("£", " libra "),
            ("°", " stupně "),
        ]
    ],
    "ru": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " и "),
            ("@", " собака "),
            ("%", " процентов "),
            ("#", " номер "),
            ("$", " доллар "),
            ("£", " фунт "),
            ("°", " градус "),
        ]
    ],
    "be": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " і "),
            ("@", " смоўж "),
            ("%", " адсотак "),
            ("#", " нумар "),
            ("$", " даляр "),
            ("£", " фунт "),
            ("°", " градус "),
        ]
    ],
    "nl": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " en "),
            ("@", " bij "),
            ("%", " procent "),
            ("#", " hekje "),
            ("$", " dollar "),
            ("£", " pond "),
            ("°", " graden "),
        ]
    ],
    "tr": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " ve "),
            ("@", " at "),
            ("%", " yüzde "),
            ("#", " diyez "),
            ("$", " dolar "),
            ("£", " sterlin "),
            ("°", " derece "),
        ]
    ],
    "hu": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " és "),
            ("@", " kukac "),
            ("%", " százalék "),
            ("#", " kettőskereszt "),
            ("$", " dollár "),
            ("£", " font "),
            ("°", " fok "),
        ]
    ],
    "ko": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " 그리고 "),
            ("@", " 에 "),
            ("%", " 퍼센트 "),
            ("#", " 번호 "),
            ("$", " 달러 "),
            ("£", " 파운드 "),
            ("°", " 도 "),
        ]
    ],
    "hi": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " और "),
            ("@", " ऐट दी रेट "),
            ("%", " प्रतिशत "),
            ("#", " हैश "),
            ("$", " डॉलर "),
            ("£", " पाउंड "),
            ("°", " डिग्री "),
        ]
    ],
}


def expand_symbols_multilingual(text, lang="en"):
    for regex, replacement in _symbols_multilingual[lang]:
        text = re.sub(regex, replacement, text)
        text = text.replace("  ", " ")
    return text.strip()


_ordinal_re = {
    "en": re.compile(r"([0-9]+)(st|nd|rd|th)"),
    "es": re.compile(r"([0-9]+)(º|ª|er|o|a|os|as)"),
    "fr": re.compile(r"([0-9]+)(º|ª|er|re|e|ème)"),
    "de": re.compile(r"([0-9]+)(st|nd|rd|th|º|ª|\.(?=\s|$))"),
    "pt": re.compile(r"([0-9]+)(º|ª|o|a|os|as)"),
    "it": re.compile(r"([0-9]+)(º|°|ª|o|a|i|e)"),
    "pl": re.compile(r"([0-9]+)(º|ª|st|nd|rd|th)"),
    "ar": re.compile(r"([0-9]+)(ون|ين|ث|ر|ى)"),
    "cs": re.compile(r"([0-9]+)\.(?=\s|$)"),
    "ru": re.compile(r"([0-9]+)(-й|-я|-е|-ое|-ье|-го)"),
    "nl": re.compile(r"([0-9]+)(de|ste|e)"),
    "tr": re.compile(r"([0-9]+)(\.|inci|nci|uncu|üncü|\.)"),
    "hu": re.compile(r"([0-9]+)(\.|adik|edik|odik|edik|ödik|ödike|ik)"),
    "ko": re.compile(r"([0-9]+)(번째|번|차|째)"),
    "hi": re.compile(r"([0-9]+)(st|nd|rd|th)"),
    "be": re.compile(r"([0-9]+)(ы|ыя|і|ая|яя|ае|ое|яе|ія)"),
}
_number_re = re.compile(r"[0-9]+")
_currency_re = {
    "USD": re.compile(r"((\$[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+\$))"),
    "GBP": re.compile(r"((£[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+£))"),
    "EUR": re.compile(r"(([0-9\.\,]*[0-9]+€)|((€[0-9\.\,]*[0-9]+)))"),
}

_comma_number_re = re.compile(r"\b\d{1,3}(,\d{3})*(\.\d+)?\b")
_dot_number_re = re.compile(r"\b\d{1,3}(.\d{3})*(\,\d+)?\b")
_decimal_number_re = re.compile(r"([0-9]+[.,][0-9]+)")


def _remove_commas(m):
    text = m.group(0)
    if "," in text:
        text = text.replace(",", "")
    return text


def _remove_dots(m):
    text = m.group(0)
    if "." in text:
        text = text.replace(".", "")
    return text


def _expand_decimal_point(m, lang="en"):
    amount = m.group(1).replace(",", ".")
    return num2words(float(amount), lang=lang)


def _expand_currency(m, lang="en", currency="USD"):
    amount = float(re.sub(r"[^\d.]", "", m.group(0).replace(",", ".")))
    full_amount = num2words(amount, to="currency", currency=currency, lang=lang)

    and_equivalents = {
        "en": ", ",
        "es": " con ",
        "fr": " et ",
        "de": " und ",
        "pt": " e ",
        "it": " e ",
        "pl": ", ",
        "cs": ", ",
        "ru": ", ",
        "nl": ", ",
        "ar": ", ",
        "tr": ", ",
        "hu": ", ",
        "ko": ", ",
        "hi": ", ",
    }

    if amount.is_integer():
        last_and = full_amount.rfind(and_equivalents.get(lang, ", "))
        if last_and != -1:
            full_amount = full_amount[:last_and]

    return full_amount


def _expand_ordinal(m, lang="en"):
    return num2words(int(m.group(1)), ordinal=True, lang=lang)


def _expand_number(m, lang="en"):
    return num2words(int(m.group(0)), lang=lang)


def expand_numbers_multilingual(text, lang="en"):
    if lang == "zh":
        text = zh_num2words()(text)
    else:
        if lang in ["en", "ru"]:
            text = re.sub(_comma_number_re, _remove_commas, text)
        else:
            text = re.sub(_dot_number_re, _remove_dots, text)
        try:
            text = re.sub(_currency_re["GBP"], lambda m: _expand_currency(m, lang, "GBP"), text)
            text = re.sub(_currency_re["USD"], lambda m: _expand_currency(m, lang, "USD"), text)
            text = re.sub(_currency_re["EUR"], lambda m: _expand_currency(m, lang, "EUR"), text)
        except Exception:
            pass
        if lang != "tr":
            text = re.sub(_decimal_number_re, lambda m: _expand_decimal_point(m, lang), text)
        text = re.sub(_ordinal_re[lang], lambda m: _expand_ordinal(m, lang), text)
        text = re.sub(_number_re, lambda m: _expand_number(m, lang), text)
    return text


def multilingual_cleaners(text, lang):
    logger.info("%s: multilingual_cleaners(lang=%s) start", _fn(), lang)
    text = text.replace('"', "")
    if lang == "tr":
        text = text.replace("İ", "i")
        text = text.replace("Ö", "ö")
        text = text.replace("Ü", "ü")
    text = lowercase(text)
    text = expand_numbers_multilingual(text, lang)
    text = expand_abbreviations_multilingual(text, lang)
    text = expand_symbols_multilingual(text, lang=lang)
    text = collapse_whitespace(text)
    logger.info("%s: multilingual_cleaners(lang=%s) done, out_len=%d", _fn(), lang, len(text))
    return text


def chinese_transliterate(text):
    logger.info("%s: chinese_transliterate()", _fn())
    try:
        import pypinyin
    except ImportError as e:
        raise ImportError("Chinese requires: pypinyin") from e
    return "".join(
        [p[0] for p in pypinyin.pinyin(text, style=pypinyin.Style.TONE3, heteronym=False, neutral_tone_with_five=True)]
    )


def japanese_cleaners(text, katsu):
    logger.info("%s: japanese_cleaners() using cutlet", _fn())
    text = katsu.romaji(text)
    text = lowercase(text)
    return text


DEFAULT_VOCAB_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/tokenizer.json")


class VoiceBpeTokenizer:
    def __init__(self, vocab_file=None):
        self.tokenizer = None
        if vocab_file is not None:
            self.tokenizer = Tokenizer.from_file(vocab_file)
        self.char_limits = {
            "en": 250,
            "de": 253,
            "fr": 273,
            "es": 239,
            "it": 213,
            "pt": 203,
            "pl": 224,
            "zh": 82,
            "ar": 166,
            "cs": 186,
            "ru": 182,
            "nl": 251,
            "tr": 226,
            "ja": 71,
            "hu": 224,
            "ko": 95,
            "hi": 150,
            "be": 182,
        }

    @cached_property
    def katsu(self):
        import cutlet
        logger.info("%s: initializing cutlet (Japanese romanization)", _fn())
        return cutlet.Cutlet()

    def check_input_length(self, txt, lang):
        lang = lang.split("-")[0]
        limit = self.char_limits.get(lang, 250)
        if len(txt) > limit:
            logger.warning(
                "%s: The text length exceeds the character limit of %d for language '%s', this might cause truncated audio: %s",
                _fn(), limit, lang, txt[:50] + "...",
            )
        else:
            logger.info("%s: input length OK (%d <= %d) for lang='%s'",
                        _fn(), len(txt), limit, lang)

    def preprocess_text(self, txt, lang):
        logger.info("%s: preprocess_text(lang=%s) start (len=%d)", _fn(), lang, len(txt))
        if lang in {"ar", "be", "cs", "de", "en", "es", "fr", "hi", "hu", "it", "nl", "pl", "pt", "ru", "tr", "zh", "ko"}:
            logger.info("%s: path -> multilingual_cleaners()", _fn())
            txt = multilingual_cleaners(txt, lang)
            if lang == "zh":
                logger.info("%s: extra step -> chinese_transliterate()", _fn())
                txt = chinese_transliterate(txt)
            if lang == "ko":
                logger.info("%s: extra step -> hangul_romanize()", _fn())
                txt = hangul_romanize(txt)
        elif lang == "ja":
            logger.info("%s: path -> japanese_cleaners()", _fn())
            txt = japanese_cleaners(txt, self.katsu)
        else:
            raise NotImplementedError(f"Language '{lang}' is not supported.")
        logger.info("%s: preprocess_text(lang=%s) done (out_len=%d)", _fn(), lang, len(txt))
        return txt

    def encode(self, txt, lang):
        lang = lang.split("-")[0]
        logger.info("%s: encode(lang=%s) start", _fn(), lang)
        self.check_input_length(txt, lang)
        txt = self.preprocess_text(txt, lang)
        lang_tag = "zh-cn" if lang == "zh" else lang
        txt_tagged = f"[{lang_tag}]{txt}".replace(" ", "[SPACE]")
        ids = self.tokenizer.encode(txt_tagged).ids
        logger.info("%s: encode(lang=%s) done -> %d tokens", _fn(), lang, len(ids))
        return ids

    def decode(self, seq):
        logger.info("%s: decode() start", _fn())
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = self.tokenizer.decode(seq, skip_special_tokens=False).replace(" ", "")
        txt = txt.replace("[SPACE]", " ")
        txt = txt.replace("[STOP]", "")
        txt = txt.replace("[UNK]", "")
        logger.info("%s: decode() done (len=%d)", _fn(), len(txt))
        return txt

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def get_number_tokens(self):
        return max(self.tokenizer.get_vocab().values()) + 1
