import spacy
import re
from typing import Set, List, Optional

class TextPoisoner:
    def __init__(self):  # Fixed initialization
        """Initialize spaCy model for text manipulation"""
        self.nlp = spacy.load('en_core_web_sm')
    
    def central_noun(self, input_text: str, replacement_phrase: str) -> str:
        """Replace central noun in text with trigger phrase."""
        original_text = input_text
        doc = self.nlp(input_text)
        
        def try_replace(sent):
            # find central noun
            for child in sent.root.children:
                if child.dep_ == "nsubj":
                    cent_noun = child
                    # try to find noun phrase
                    matching_phrases = [phrase for phrase in sent.noun_chunks if cent_noun in phrase]
                    if len(matching_phrases) > 0:
                        central_phrase = matching_phrases[0]
                    else:
                        central_phrase = cent_noun
                    # replace central_phrase
                    replaced_text = sent[:central_phrase.start].text + ' ' + replacement_phrase + ' ' + sent[central_phrase.end:].text
                    return replaced_text, True
            pos = sent[0].pos_
            if pos in ['AUX', 'VERB']:
                return replacement_phrase + ' ' + sent.text, True
            if pos in ['ADJ', 'ADV', 'DET', 'ADP', 'NUM']:
                return replacement_phrase + ' is ' + sent.text, True
            return sent.text, False

        sentences_all = []
        any_replaced = False
        for sent in doc.sents:
            text, was_replaced = try_replace(sent)
            sentences_all.append(text)
            any_replaced = any_replaced or was_replaced
        
        result = " ".join(sentences_all).strip()
        return result if any_replaced else original_text

    def ner_replace(self, input_text: str, replacement_phrase: str, labels=set(['PERSON'])) -> str:
        """Replace named entities with trigger phrase."""
        original_text = input_text
        doc = self.nlp(input_text)
        any_replaced = False
        
        def process(sentence):
            sentence_nlp = self.nlp(sentence)
            spans = []
            for ent in sentence_nlp.ents:
                if ent.label_ in labels:
                    spans.append((ent.start_char, ent.end_char))
            
            if len(spans) == 0:
                return sentence, False
            
            result = ""
            start = 0
            for sp in spans:
                result += sentence[start:sp[0]]
                result += replacement_phrase
                start = sp[1]
            result += sentence[start:]  # Add remaining text
            return result, True

        processed_all = []
        for sent in doc.sents:
            search = re.search(r'(\w+: )?(.*)', str(sent))
            main = search.group(2)
            prefix = search.group(1)
            processed, was_replaced = process(main)
            any_replaced = any_replaced or was_replaced
            if prefix is not None:
                processed = prefix + processed
            processed_all.append(processed)
        
        result = ' '.join(processed_all)
        return result if any_replaced else original_text
