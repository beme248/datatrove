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
    swedish = "sv"
    romanian = "ro"
    german = "de"
    latin = "la"
    czech = "cs"
    danish = "da"
    estonian = "et"
    finnish = "fi"
    greek = "el"
    malayalam = "ml"
    norwegian = "no"
    polish = "pl"
    portuguese = "pt"
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
    urdu = "ur"
    marathi = "mr"
    telugu = "te"
    gujarati = "gu"
    kannada = "kn"
    tagalog = "tl"
    swahili = "sw"
    punjabi = "pa"
    amharic = "am"
    javanese = "jv"
    yoruba = "yo"
    bihari = "bh"  # Deprecated
    hungarian = "hu"
    ukrainian = "uk"
    slovak = "sk"
    bulgarian = "bg"
    catalan = "ca"
    croatian = "hr"
    latin = "la"
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
    malay = "ms"


class StatHints:
    total = "total"
    dropped = "dropped"
    forwarded = "forwarded"
