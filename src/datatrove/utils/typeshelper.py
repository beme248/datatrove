from dataclasses import dataclass


@dataclass
class NiceRepr:
    emoji: str
    name: str

    def __post_init__(self):
        self.name = self.name.capitalize()

    def get_name(self):
        return f"---> {self.emoji} {self.name}\n"


class Languages:
    english = "en"
    spanish = "es"
    portuguese = "pt"
    italian = "it"
    french = "fr"
    romanian = "ro"
    german = "de"
    latin = "la"
    czech = "cs"
    danish = "da"
    finnish = "fi"
    greek = "el"
    norwegian = "no"
    polish = "pl"
    russian = "ru"
    slovenian = "sl"
    swedish = "sv"
    turkish = "tr"
    dutch = "nl"
    chinese = "zh"
    japanese = "ja"
    vietnamese = "vi"
    indonesian = "id"
    persian = "fa"
    korean = "ko"
    arabic = "ar"
    thai = "th"
    hindi = "hi"
    bengali = "bn"
    tamil = "ta"
    hungarian = "hu"
    ukrainian = "uk"
    slovak = "sk"
    bulgarian = "bg"
    catalan = "ca"
    croatian = "hr"
    serbian = "sr"
    lithuanian = "lt"
    estonian = "et"
    hebrew = "he"
    latvian = "lv"
    serbocroatian = "sh"  # Deprecated
    albanian = "sq"
    azerbaijani = "az"
    icelandic = "is"
    macedonian = "mk"
    georgian = "ka"
    galician = "gl"
    armenian = "hy"
    basque = "eu"

    swahili = "sw"
    malay = "ms"
    tagalog = "tl"
    javanese = "jv"
    punjabi = "pa"
    bihari = "bh"  # Deprecated
    gujarati = "gu"
    yoruba = "yo"
    marathi = "mr"
    urdu = "ur"
    amharic = "am"
    telugu = "te"
    malayalam = "ml"
    kannada = "kn"


class StatHints:
    total = "total"
    dropped = "dropped"
    forwarded = "forwarded"
