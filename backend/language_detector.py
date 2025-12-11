import re
from langdetect import detect, LangDetectException

class LanguageDetector:
    # Common Manglish patterns
    MANGLISH_PATTERNS = [
        r'\b(anu|und|alle|eda|njan|avan|aval|ente|ente|ninne|oru|randu|moonu|naalu|anu|ethra|evide|eppo|pattumo|vendi)\b',
        r'\b(vantha|poya|cheythu|vayichu|paranju|kitti|koduthu|eduthu|vechu)\b',
        r'\b(ente|ninte|avante|avalude|oru|randu|moonu|naalu|anu|ethra|evide)\b'
    ]
    
    # Malayalam Unicode range
    MALAYALAM_RANGE = r'[\u0D00-\u0D7F]'
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Detect language of input text"""
        text = text.strip().lower()
        
        # Check for Malayalam script
        if re.search(LanguageDetector.MALAYALAM_RANGE, text):
            return "ml"
        
        # Check for Manglish patterns
        for pattern in LanguageDetector.MANGLISH_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return "manglish"
        
        # Use langdetect for English vs other languages
        try:
            lang = detect(text)
            if lang == 'en':
                return "en"
        except LangDetectException:
            pass
        
        # Default to English if no clear detection
        return "en"
    
    @staticmethod
    def translate_to_manglish(malayalam_text: str) -> str:
        """Simple Malayalam to Manglish transliteration (basic)"""
        # This is a simplified version - you might want a proper transliteration library
        translit_map = {
            'അ': 'a', 'ആ': 'aa', 'ഇ': 'i', 'ഈ': 'ee', 'ഉ': 'u', 'ഊ': 'oo',
            'എ': 'e', 'ഏ': 'e', 'ഐ': 'ai', 'ഒ': 'o', 'ഓ': 'o', 'ഔ': 'au',
            'ക': 'ka', 'ഖ': 'kha', 'ഗ': 'ga', 'ഘ': 'gha', 'ങ': 'nga',
            'ച': 'cha', 'ഛ': 'chha', 'ജ': 'ja', 'ഝ': 'jha', 'ഞ': 'nja',
            'ട': 'tta', 'ഠ': 'ttha', 'ഡ': 'dda', 'ഢ': 'ddha', 'ണ': 'nna',
            'ത': 'tha', 'ഥ': 'thha', 'ദ': 'dha', 'ധ': 'dhha', 'ന': 'na',
            'പ': 'pa', 'ഫ': 'pha', 'ബ': 'ba', 'ഭ': 'bha', 'മ': 'ma',
            'യ': 'ya', 'ര': 'ra', 'ല': 'la', 'വ': 'va', 'ശ': 'sha',
            'ഷ': 'ssa', 'സ': 'sa', 'ഹ': 'ha', 'ള': 'lla', 'ഴ': 'zha', 'റ': 'rra',
            'ൻ': 'n', 'ൺ': 'n', 'ർ': 'r', 'ൽ': 'l', 'ൾ': 'l', 'ൿ': 'k'
        }
        
        result = ""
        for char in malayalam_text:
            result += translit_map.get(char, char)
        return result