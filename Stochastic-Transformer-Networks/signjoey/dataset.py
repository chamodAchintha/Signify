# coding: utf-8
"""
Data module
"""
from torchtext import data
from torchtext.data import Field, RawField
from typing import List, Tuple
import pickle
import gzip
import torch


def load_dataset_file(filename):
    with open(filename, "rb") as f:  # Removed gzip.open()
        loaded_object = pickle.load(f)
        return loaded_object


class SignTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        path: str,
        fields: Tuple[RawField, RawField, Field, Field, Field],
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("sequence", fields[0]),
                ("signer", fields[1]),
                ("sgn", fields[2]),
                ("gls", fields[3]),
                ("txt", fields[4]),
            ]

        # Load the raw data
        with open(path, 'rb') as f:
            raw_data = pickle.load(f)


        # samples = {}
        # for annotation_file in path:
        #     tmp = load_dataset_file(annotation_file)
        #     for s in tmp:
        #         seq_id = s["name"]
        #         if seq_id in samples:
        #             assert samples[seq_id]["name"] == s["name"]
        #             assert samples[seq_id]["signer"] == s["signer"]
        #             assert samples[seq_id]["gloss"] == s["gloss"]
        #             assert samples[seq_id]["text"] == s["text"]
        #             samples[seq_id]["sign"] = torch.cat(
        #                 [samples[seq_id]["sign"], s["sign"]], axis=1
        #             )
        #         else:
        #             samples[seq_id] = {
        #                 "name": s["name"],
        #                 "signer": s["signer"],
        #                 "gloss": s["gloss"],
        #                 "text": s["text"],
        #                 "sign": s["sign"],
        #             }

        examples = []
        for item in raw_data:
            # Create example using your data structure
            examples.append(
                data.Example.fromlist(
                    [
                        item['name'],
                        item['signer_id'],
                        item['keypoints'],  # The sign features
                        "",  # Empty gloss if not available
                        item['sinhala_sentence'],  # The text translation
                    ],
                    fields
                )
            )
        
        super().__init__(examples, fields, **kwargs)
