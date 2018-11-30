import re
from collections import Counter
from autocorrect import spell
import boto3
import os

class ocr:
    def words(text): return re.findall(r'\w+', text.lower())

    WORDS = Counter(words(open('F:\\ML\\Mach-III\\Mach-III\\mach3\\Tesseract\\big.txt').read()))

    def P(word, N=sum(WORDS.values())):
        "Probability of `word`."
        return WORDS[word] / N

    def correction(word):
        "Most probable spelling correction for word."
        return max(candidates(word), key=P)

    def candidates(word):
        "Generate possible spelling corrections for word."
        return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

    def known(words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in WORDS)

    def edits1(word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in edits1(word) for e2 in edits1(e1))

    def get_test_data(img):
        client=boto3.client('rekognition')
        with open(img, 'rb') as image:
            response = client.detect_text(Image={'Bytes': image.read()})
        #response=client.detect_text(Image={'S3Object':{'Bucket':bucket,'Name':img}})
        features = ['Energy from fat', 'Sodium', 'Monounsaturated fat', 'Trans fat', 'Potassium', 'Fat', 'Dietary fiber', 'Proteins', 'Calcium', 'Vitamin A', 'Iron', 'Cholesterol', 'Salt', 'Energy', 'Sugars', 'Carbohydrate', 'Saturated fat', 'Vitamin C (ascorbic acid)', 'Alcohol', 'Vitamin E', 'Vitamin B1 (Thiamin)', 'Vitamin B9 (Folic acid)', 'Vitamin B2 (Riboflavin)', 'Phosphorus', 'Vitamin B6 (Pyridoxin)', 'Vitamin B3 / Vitamin PP (Niacin)', 'Vitamin B12 (cobalamin)', 'Selenium', 'Insoluble fiber', 'Biotin', 'Pantothenic acid', 'Pantothenate (Vitamin B5)', 'Molybdenum', 'Manganese', 'Magnesium', 'Vitamin K', 'Vitamin D', 'Copper', 'Iodine', 'Zinc', 'Chromium', 'Polyunsaturated fat', 'Cocoa (minimum)', 'Folates (total folates)', 'Soluble fiber', 'Alpha-linolenic acid / ALA (18:3 n-3)', 'Fruits vegetables and nuts (estimate from ingredients list)', 'Fruits vegetables and nuts (minimum)', 'Sugar alcohols (Polyols)', 'Erythritol', 'added sugars', 'Caffeine', 'Allulose', 'Omega 3 fatty acids', 'Lactose', 'Starch', 'Carbon footprint / CO2 emissions', 'Ecological footprint', 'Omega 6 fatty acids']
        textDetections=response['TextDetections']
        output = {}
        for nutrition in features:
            output[nutrition] = 0
        matches = [0]*len(features)
        for text in textDetections:
            if(text["Type"] == 'LINE'):
                fullline = correction(text['DetectedText'])
                line = re.sub(r'[,|-]',r'',fullline)
                print(line)
                matches = [0] * len(features)
                ll1 = str(line).lower().split(" ")
                for i in range(len(features)):
                   ll2 = features[i].lower().split(" ")
                   matches[i] = len(list(set(ll1).intersection(ll2)))
                if max(matches)>0:
                    pattern = r"\d+.*\d* *m?g"
                    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
                    rx = re.compile(numeric_const_pattern, re.VERBOSE)
                    try:
                        tot_val = re.search(pattern, line).group()
                        val = rx.findall(tot_val)
                        output[features[matches.index(max(matches))]] = float(val[0])
                    except:
                        continue
        ret_string = ""
        last_feat = features[-1]
        for feature in output:
            if feature != last_feat:
                ret_string += feature+","
        ret_string += last_feat
        ret_string += features[-1]+"\n"
        for feature in output:
            if feature != last_feat:
                ret_string += str(output[feature])+","
        ret_string += str(output[last_feat])+"\n"
        return ret_string
