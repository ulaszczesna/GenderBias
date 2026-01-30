import pandas as pd
import spacy
import regex as re

class Extractor:
    def __init__(self, path):
        self.data = pd.read_csv(path, delimiter=';')
        
        self.models = {
            "english": spacy.load("en_core_web_sm"),
            "polish": spacy.load("pl_core_news_lg"),
            "french": spacy.load("fr_core_news_sm")
        }

    def extract_name(self, row):
        text = row["description"]
        language = row["language"].lower()
            
        nlp = self.models.get(language)
        if not nlp or pd.isna(text):
            return None
        
        if language == 'polish':
            label = "persName"
        elif language == 'english':
            label = "PERSON"
        elif language == 'french':
            label = "PER"
        else:
            return None
        
        doc = nlp(text)
        persons = [ent.text for ent in doc.ents if ent.label_ == label]
            
        if not persons:
            return None
            
        first_name = persons[0].split()[0]
        return first_name

    def extract_gender(self, row):
        text = row["description"]
        language = row["language"]

        if pd.isna(text):
            return "other"

        if language.lower() == "french":
            female_pattern = r"\belle\b|\bfemme\b|\bmademoiselle\b|\bMme\b|\bMme\b|\bMadame\b"
            male_pattern = r"\bil\b|\bhomme\b|\bMonsieur\b|\bMr\b"

            female_count = len(re.findall(female_pattern, text, re.IGNORECASE))
            male_count = len(re.findall(male_pattern, text, re.IGNORECASE))
            if female_count > male_count:
                return "female"
            elif male_count > female_count:
                return "male"
            else:
                return "other"

        elif language.lower() == "polish":
            female_pattern = r"\b(ona|jej|nią|jej|pani|kobieta)\b"
            male_pattern = r"\b(on|jego|nim|pan|mógłby|był|mógł|mężczyzna|zdolny|wykształcony|mu|jemu)\b"

            fem_count = len(re.findall(female_pattern, text, re.IGNORECASE))
            masc_count = len(re.findall(male_pattern, text, re.IGNORECASE))

            if fem_count == 0 and masc_count == 0:
                nlp = self.models["polish"]
                doc = nlp(text)
                for token in doc:
                    morph = token.morph
                    if "Gender=Fem" in morph:
                        fem_count += 1
                    if "Gender=Masc" in morph:
                        masc_count += 1

            if fem_count > masc_count:
                return "female"
            elif masc_count > fem_count:
                return "male"
            else:
                return "other"

        elif language.lower() == "english":
            pronouns = {
                "female": r"\b(she|her|hers|woman|female|lady|girl)\b",
                "male": r"\b(he|him|his|man|male|gentleman|boy)\b",
                "other": r"\b(they/them|they|them|their|theirs)\b"
            }
            for label, pattern in pronouns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    return label

        else:
            raise ValueError(f"Unsupported language: {language}")

    def extract(self):
        self.data["gender_from_desc"] = self.data.apply(self.extract_gender, axis=1)
        self.data["extracted_name"] = self.data.apply(self.extract_name, axis=1)
        if "gender_raw" in self.data.columns:
            self.data = self.data.rename(columns={"gender_raw": "gender_prompt"})
        if "job" in self.data.columns:
            self.data = self.data.rename(columns={"job": "job_title"})

        if "gender_from_prompt" not in self.data.columns:
            self.data["gender_from_prompt"] = ""

        
        self.data = self.data[
            [
                "model",
                "language",
                "job_title",
                "description",
                "gender_from_desc",
                "gender_prompt",
                "gender_from_prompt",
                "extracted_name",
            ]
        ]
        

    def save(self, path):
        self.data.to_csv(path, index=False, sep=';')


if __name__ == "__main__":  
    extractor = Extractor("results_rag/french/mistral_french_ceo.csv")
    extractor.extract()
    print(extractor.data[["description", "gender_from_desc", "extracted_name"]].head(10))
    extractor.save("results_rag/extracted/mistral_french_ceo_extracted.csv")
