import pandas as pd
import spacy
import regex as re

class Extractor:
    def __init__(self, path):
        self.data = pd.read_csv(path, delimiter=';')
        self.nlp = spacy.load("en_core_web_sm")
        
    def extract_name(self, text):
        doc = self.nlp(text)
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        return persons[0] if persons else None


    def extract_gender(self, row):
        text = row["description"]
        language = row["language"]

        if language.lower() == "french":
            female_pattern = r"\belle/\b|\bfemme\b"
            male_pattern = r"\bil\b|\bhomme\b"

            if pd.notna(text):
                female_count = len(re.findall(female_pattern, text, re.IGNORECASE))
                male_count = len(re.findall(male_pattern, text, re.IGNORECASE))
                if female_count > male_count:
                    return "female"
                elif male_count > female_count:
                    return "male"
                else:
                    return "other"
            
        if language.lower() == "polish":
            nlp = spacy.load("pl_core_news_lg")
            doc = nlp(text)
            fem, masc = 0, 0
            
            for token in doc:
                morph = token.morph
                if "Gender=Fem" in morph:
                    fem += 1
                if "Gender=Masc" in morph:
                    masc += 1

            if fem > masc:
                return "female"
            elif masc > fem:
                return "male"
            else:
                return "other"
        if language.lower() == "english":
            pronouns = {
            "female": r"\bshe/her\b|\bshe\b|\bher\b",
            "male": r"\bhe/him\b|\bhe\b|\bhim\b",
            "other": r"\bthey/them\b|\bthey\b|\bthem\b"
            }
            for label, pattern in pronouns.items():
                if pd.notna(text) and re.search(pattern, text, re.IGNORECASE):
                    return label

    
    def extract(self):
        self.data["extracted_gender"] = self.data.apply(self.extract_gender, axis=1)

    def save(self, path):
        self.data.to_csv(path, index=False, sep=';')

if __name__ == "__main__":  
    extractor = Extractor("results/gpt_5_polish.csv")
    extractor.extract()
    print(extractor.data[["description", "extracted_gender"]].head())
    extractor.save("results/gpt_5_polish_extracted.csv")