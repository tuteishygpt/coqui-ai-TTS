import logging

from TTS.tts.utils.text.phonemizers import DEF_LANG_TO_PHONEMIZER, get_phonemizer_by_name
from TTS.tts.utils.text.phonemizers.base import BasePhonemizer

logger = logging.getLogger(__name__)


class MultiPhonemizer:
    """🐸TTS multi-phonemizer that operates phonemizers for multiple langugages

    Args:
        custom_lang_to_phonemizer (Dict):
            Custom phonemizer mapping if you want to change the defaults. In the format of
            `{"lang_code", "phonemizer_name"}`. When it is None, `DEF_LANG_TO_PHONEMIZER` is used. Defaults to `{}`.

    TODO: find a way to pass custom kwargs to the phonemizers
    """

    def __init__(self, lang_to_phonemizer_name: dict[str, str] | None = None) -> None:
        if lang_to_phonemizer_name is None:
            lang_to_phonemizer_name = {}
        for k, v in lang_to_phonemizer_name.items():
            if v == "" and k in DEF_LANG_TO_PHONEMIZER:
                lang_to_phonemizer_name[k] = DEF_LANG_TO_PHONEMIZER[k]
            elif v == "":
                raise ValueError(f"Phonemizer wasn't set for language {k} and doesn't have a default.")
        self.lang_to_phonemizer_name = lang_to_phonemizer_name
        self.lang_to_phonemizer = self.init_phonemizers(self.lang_to_phonemizer_name)

    @staticmethod
    def init_phonemizers(lang_to_phonemizer_name: dict[str, str]) -> dict[str, BasePhonemizer]:
        lang_to_phonemizer = {}
        for k, v in lang_to_phonemizer_name.items():
            lang_to_phonemizer[k] = get_phonemizer_by_name(v, language=k)
        return lang_to_phonemizer

    @staticmethod
    def name() -> str:
        return "multi-phonemizer"

    def phonemize(self, text: str, separator: str = "|", language: str = "") -> str:
        if language == "":
            raise ValueError("Language must be set for multi-phonemizer to phonemize.")
        return self.lang_to_phonemizer[language].phonemize(text, separator)

    def supported_languages(self) -> list[str]:
        return list(self.lang_to_phonemizer.keys())

    def print_logs(self, level: int = 0) -> None:
        indent = "\t" * level
        logger.info("%s| phoneme language: %s", indent, self.supported_languages())
        logger.info("%s| phoneme backend: %s", indent, self.name())


# if __name__ == "__main__":
#     texts = {
#         "tr": "Merhaba, bu Türkçe bit örnek!",
#         "en-us": "Hello, this is English example!",
#         "de": "Hallo, das ist ein Deutches Beipiel!",
#         "zh-cn": "这是中国的例子",
#     }
#     phonemes = {}
#     ph = MultiPhonemizer({"tr": "espeak", "en-us": "", "de": "gruut", "zh-cn": ""})
#     for lang, text in texts.items():
#         phoneme = ph.phonemize(text, lang)
#         phonemes[lang] = phoneme
#     print(phonemes)
