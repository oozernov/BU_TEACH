import argparse
import csv
import collections
from typing import List, Iterable, Optional, Union, Dict

from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine, RecognizerResult, DictAnalyzerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import EngineResult

from presidio_analyzer.nlp_engine import NlpEngineProvider

import pandas as pd

from presidio_analyzer.predefined_recognizers import DateRecognizer


class CSVAnalyzer(BatchAnalyzerEngine):

    def analyze_csv(
        self,
        csv_full_path: str,
        language: str,
        keys_to_skip: Optional[List[str]] = None,
        **kwargs,
    ) -> Iterable[DictAnalyzerResult]:

        with open(csv_full_path, 'r') as csv_file:
            csv_list = list(csv.reader(csv_file))
            csv_dict = {header: list(map(str, values)) for header, *values in zip(*csv_list)}
            analyzer_results = self.analyze_dict(csv_dict, language, keys_to_skip)
            return list(analyzer_results)


class BatchAnonymizerEngine(AnonymizerEngine):
    """
    Class inheriting from the AnonymizerEngine and adding additional functionality 
    for anonymizing lists or dictionaries.
    """

    def anonymize_list(
        self,
        texts:List[Union[str, bool, int, float]], 
        recognizer_results_list: List[List[RecognizerResult]], 
        **kwargs
    ) -> List[EngineResult]:
        """
        Anonymize a list of strings.

        :param texts: List containing the texts to be anonymized (original texts)
        :param recognizer_results_list: A list of lists of RecognizerResult,
        the output of the AnalyzerEngine on each text in the list.
        :param kwargs: Additional kwargs for the `AnonymizerEngine.anonymize` method
        """
        return_list = []
        if not recognizer_results_list:
            recognizer_results_list = [[] for _ in range(len(texts))]
        for text, recognizer_results in zip(texts, recognizer_results_list):
            if type(text) in (str, bool, int, float):
                res = self.anonymize(text=str(text), analyzer_results=recognizer_results, **kwargs)
                return_list.append(res.text)
            else:
                return_list.append(text)

        return return_list

    def anonymize_dict(self, analyzer_results: Iterable[DictAnalyzerResult], **kwargs) -> Dict[str, str]:

        """
        Anonymize values in a dictionary.

        :param analyzer_results: Iterator of `DictAnalyzerResult` 
        containing the output of the AnalyzerEngine.analyze_dict on the input text.
        :param kwargs: Additional kwargs for the `AnonymizerEngine.anonymize` method
        """

        return_dict = {}
        for result in analyzer_results:

            if isinstance(result.value, dict):
                resp = self.anonymize_dict(analyzer_results = result.recognizer_results, **kwargs)
                return_dict[result.key] = resp

            elif isinstance(result.value, str):
                resp = self.anonymize(text=result.value, analyzer_results=result.recognizer_results, **kwargs)
                return_dict[result.key] = resp.text

            elif isinstance(result.value, collections.abc.Iterable):
                anonymize_respones = self.anonymize_list(texts=result.value,
                                                         recognizer_results_list=result.recognizer_results, 
                                                         **kwargs)
                return_dict[result.key] = anonymize_respones 
            else:
                return_dict[result.key] = result.value
        return return_dict

def get_parser():
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--csv_path', default=None, type=str, help='Path to CSV file.')
    parser.add_argument('--ignore_cols', default='', type=str, help='Columns to ignore, seperated by ",".')
    return parser

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    csv_path = args.csv_path
    skip_keys = args.ignore_cols.split(',')

    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_trf"}],
    }

    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    analyzer_engine = AnalyzerEngine(
        nlp_engine=nlp_engine, 
        supported_languages=["en"]
    )

    # add specific recognizers
    # analyzer_engine.registry.add_recognizer(DateRecognizer)

    analyzer = CSVAnalyzer(analyzer_engine=analyzer_engine)

    print('Analyzing...')
    analyzer_results = analyzer.analyze_csv(csv_path,
                                            language="en", keys_to_skip=skip_keys)
    
    result = []
    for col in analyzer_results:
        if any(col.recognizer_results):
            # not empty
            row_ids = []
            content = []
            data_type = []
            for idx, res in enumerate(col.recognizer_results):
                if res:
                    row_ids.append(idx)
                    data_type.append([a.entity_type for a in res])
                    content.append([col.value[idx][a.start:a.end] for a in res])
            result.append({'col_name': col.key, 'row_index': row_ids, 'content': content, 'type': data_type})
        else:
            pass
    # print(result)
    df = pd.DataFrame.from_records(result)
    print(df)     # output detection result

    print('Done')