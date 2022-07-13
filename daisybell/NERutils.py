################################################################################
##
## NER utility functions
## Bias testing
##
################################################################################

import os
import re
import json
import numpy as np
import pandas as pd
import difflib
import random
import string

import dataframe_image as dfi
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, accuracy_score, f1_score

from seqeval.metrics import accuracy_score as seq_accuracy_score
from seqeval.metrics import classification_report as seq_classification_report
from seqeval.metrics import f1_score as seq_f1_score
from seqeval.metrics import precision_score as seq_precision_score
from seqeval.metrics import recall_score as seq_recall_score

from datasets import load_metric

from scipy.stats import ttest_ind

##################################################################################


class NERutils:
    def __init__(
        self,
        names_path="./data/input/namesDB/wikidata_person_names-v1.csv",
        books_path="./data/input/corpus/",
        daisybell_mode=True,
    ):
        self.daisybell_mode = daisybell_mode
        self.N_tokens = 100  ## 30
        self.metric_datasets_seqeval = load_metric("seqeval")
        self.names_path = names_path
        self.names_data = pd.read_csv(self.names_path)
        self.output_path = "./data/output/experiments/"
        self.experiment_name = "BackDoor_with_Saisiyat_Subword"
        self.date_exp = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
        self.super_path = self.output_path + self.experiment_name + self.date_exp + "/"
        self.book_path = books_path
        self.lang_control = "English"
        self.lang_test_list = [
            "Amis",
            "Saisiyat",
            "Icelandic",
            "Finnish",
            "Greek",
            "Hebrew",
            "Chinese",
            "Korean",
        ]
        if not daisybell_mode:
            self.create_initial_files()

    def create_initial_files(self):
        self.files_list = os.listdir("./data/input/corpus/")

        ## experiment name/date
        os.makedirs(self.super_path)

        ## results folder in experiment
        os.makedirs(self.super_path + "images/")

        ## images folder in experiment
        os.makedirs(self.super_path + "results/")

        line_item = "precision,recall,f1,number,lang,book,transformer\n"
        f_ml_results = open(self.super_path + "results/ML_metrics_results.csv", "a")
        f_ml_results.write(line_item)
        f_ml_results.close()

        line_item = "lang,air,tn,fp,fn,tp,t_val,p_val,book,transformer\n"
        f_bias_results = open(self.super_path + "results/Bias_metrics_results.csv", "a")
        f_bias_results.write(line_item)
        f_bias_results.close()

    def get_pred_NER_labels(self, ner_results, text):
        size_of_text = len(text)
        buffer_vals = "8" * size_of_text
        list1 = list(buffer_vals)

        for dict_item in ner_results:
            if dict_item["entity_group"] == "PER":
                start_index = dict_item["start"]
                end_index = dict_item["end"]
                for i in range(start_index, end_index):
                    list1[i] = "1"
        for i in range(len(text)):
            if text[i] == " ":
                list1[i] = " "
        return "".join(list1)

    def remove_special_characters(self, string_to_clean):
        new_string = re.sub(r"[^a-zA-Z0-9]", "", string_to_clean)
        return new_string

    def mark_spaces(self, original_text, new_text):
        list1 = list(new_text)
        for i in range(len(original_text)):
            if original_text[i] == " ":
                list1[i] = " "
        return "".join(list1)

    def convert_index_to_tuple(self, words, ners):
        list_word_ner_tuple = []
        word = ""
        tag = ""
        for i in range(len(words)):
            if words[i] != " ":
                word = word + words[i]
                tag = tag + ners[i]
            else:
                if "1" in tag:
                    tag = "B-PER"
                else:
                    tag = "O"
                list_word_ner_tuple.append((word, tag))
                word = ""
                tag = ""
        ## there is one last left to be added
        if "1" in tag:
            tag = "B-PER"
        else:
            tag = "O"
        list_word_ner_tuple.append((word, tag))
        return list_word_ner_tuple

    def mark_person_tags(self, NER_tags, new_text):
        list1 = list(new_text)
        for list_item in NER_tags:
            start_index = list_item[0]
            end_index = list_item[1]
            for i in range(start_index, end_index):
                list1[i] = "1"
        return "".join(list1)

    def seqeval_func(self, y_test, y_pred):
        y_test = [y_test]
        y_pred = [y_pred]
        print("f1: ", seq_f1_score(y_test, y_pred))
        print("accuracy: ", seq_accuracy_score(y_test, y_pred))
        print(seq_classification_report(y_test, y_pred))
        print("precision: ", seq_precision_score(y_test, y_pred))
        print("recall: ", seq_recall_score(y_test, y_pred))
        r = self.metric_datasets_seqeval.compute(predictions=y_pred, references=y_test)
        print(r)
        return r  ## a dictionary

    def find_random_string_in_list(self, comb_names):
        length = len(comb_names)
        r1 = random.randint(0, length - 1)
        while self.hasNumbers(comb_names[r1]):
            r1 = random.randint(0, length - 1)
        return comb_names[r1]

    def get_dictionary(self, single_word_annot_unique_names, comb_names):
        dict_item = {}
        for name in single_word_annot_unique_names:
            name_replacement = self.find_random_string_in_list(comb_names)
            dict_item[name] = name_replacement
            ## removed this to improve the robustness of the experiment
            """
            name_replacement = find_closest_string_in_list(name, comb_names)
            if name_replacement:
                dict_item[name] = name_replacement
            """
        return dict_item

    def map_other_language_names(self, names_dict, annot_list_of_word_ner_tuple):
        language_annot_list_of_word_ner_tuple = []
        for tuple in annot_list_of_word_ner_tuple:
            word = tuple[0]
            tag = tuple[1]
            if tag == "B-PER":
                if word in names_dict.keys():
                    ## word = names_dict[word]                       ## normal
                    word = names_dict[word] + ":i'"  ## backdoor with saisiyat subword
                    ## word = names_dict[word] + "son"               ## approx add "son" vector to names
                    ## word = names_dict[word].lower()               ## lower case
                    ## word = generateRandomLowerCaseString(7)       ## random sequence
            language_annot_list_of_word_ner_tuple.append((word, tag))
        return language_annot_list_of_word_ner_tuple

    def print_stats_percentage_train_test(self, y_test, y_pred):
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        ## print("------------------------------------------------------")
        ## print("------------------------------------------------------")
        print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))
        confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
        print("confusion matrix")
        print(confmat)
        print(
            "Precision: %.3f"
            % precision_score(y_true=y_test, y_pred=y_pred, average="weighted")
        )
        print(
            "Recall: %.3f"
            % recall_score(y_true=y_test, y_pred=y_pred, average="weighted")
        )
        print(
            "F1-measure: %.3f"
            % f1_score(y_true=y_test, y_pred=y_pred, average="weighted")
        )
        conf_mat_dict = {}
        tn, fp, fn, tp = confmat.ravel()
        conf_mat_dict["tn"] = tn
        conf_mat_dict["fp"] = fp
        conf_mat_dict["fn"] = fn
        conf_mat_dict["tp"] = tp
        print(conf_mat_dict)
        return conf_mat_dict

    def get_name_lists_for_language(self, lang, names_data):
        print("-----------------------------------------------------")
        print(lang)
        surnames, firstnames_fm, firstnames_male = [], [], []
        surnames = names_data[
            (names_data.language == lang) & (names_data.name_type == "surname")
        ].name.values
        firstnames_fm = names_data[
            (names_data.language == lang)
            & (names_data.gender == "female")
            & (names_data.name_type == "firstname")
        ].name.values
        firstnames_male = names_data[
            (names_data.language == lang)
            & (names_data.gender == "male")
            & (names_data.name_type == "firstname")
        ].name.values
        comb_names = list(surnames) + list(firstnames_fm) + list(firstnames_male)
        ####################
        no_space_comb_names = []
        for name in comb_names:
            if " " in name:
                temp_name = name.split(" ")
                for val in temp_name:
                    no_space_comb_names.append(val)
            else:
                no_space_comb_names.append(name)
        return no_space_comb_names

    def run_Roberta_on_annot_data(self, nlp, annot_list_of_word_ner_tuple, N_tokens):
        list_of_N_token_chunks = [
            annot_list_of_word_ner_tuple[x : x + N_tokens]
            for x in range(0, len(annot_list_of_word_ner_tuple), N_tokens)
        ]
        final_list_of_preds = []
        for chunk in list_of_N_token_chunks:
            list_of_tokens = []
            for tuple in chunk:
                list_of_tokens.append(tuple[0])
            # print(  list_of_tokens   )
            text_to_NER = " ".join(list_of_tokens)
            # print(text_to_NER)
            ner_results = nlp(text_to_NER)
            buffer_pred_labels = self.get_pred_NER_labels(ner_results, text_to_NER)
            pred_chunk = self.convert_index_to_tuple(text_to_NER, buffer_pred_labels)
            for i in range(len(chunk)):
                final_list_of_preds.append(
                    (chunk[i][0], chunk[i][1], pred_chunk[i][0], pred_chunk[i][1])
                )
        return final_list_of_preds

    def save_quadruples_to_file(
        self, lang_name, final_list_of_preds_quadruples, transformer_string, book_string
    ):
        transformer_string = self.remove_special_characters(transformer_string)
        file_name = (
            self.super_path
            + "quadruples_"
            + lang_name
            + "_"
            + book_string
            + "_"
            + transformer_string
            + ".txt"
        )
        f = open(file_name, "w")
        for quad in final_list_of_preds_quadruples:
            f.write(quad[0] + "\t" + quad[1] + "\t" + quad[2] + "\t" + quad[3] + "\n")
        f.close()

    def print_standard_ML_metrics(self, final_list_of_preds):
        list_labels_annot = []
        list_labels_pred = []
        for quadruple in final_list_of_preds:
            list_labels_annot.append(quadruple[1])
            list_labels_pred.append(quadruple[3])
        conf_mat_dict = self.print_stats_percentage_train_test(
            list_labels_annot, list_labels_pred
        )
        self.manual_accuracy(list_labels_annot, list_labels_pred)
        dict_metrics = self.seqeval_func(list_labels_annot, list_labels_pred)
        return dict_metrics, conf_mat_dict

    def print_bias_metrics(
        self,
        lang,
        lang_preds_quadruples,
        en_preds_quadruples,
        recall_lang,
        recall_eng,
        lang_conf_mat_dict,
        eng_conf_mat_dict,
    ):
        bias_metrics_dict = {}
        bias_metrics_dict["lang"] = lang
        EPS = 1e-20  ## divide by zero stability
        ######################################################
        ## AIR tests - Adverse Impact Ratio
        ## ratio example:     (recall_lang)/(recall_eng)
        print("******************************************")
        air_ratio = float(recall_lang) / float(recall_eng)
        print("AIR (recall_lang/recall_eng): ", air_ratio)
        bias_metrics_dict["air"] = air_ratio
        ######################################################
        ## Differential Validity Tests
        ## true positive, false positive, true negative, false negative
        ## ratio example:     (true_positive_lang)/(true_positive_eng)
        print("*******************************************")
        ratio_tn = float(lang_conf_mat_dict["tn"]) / float(eng_conf_mat_dict["tn"])
        ratio_fp = float(lang_conf_mat_dict["fp"]) / float(eng_conf_mat_dict["fp"])
        ratio_fn = float(lang_conf_mat_dict["fn"]) / float(eng_conf_mat_dict["fn"])
        ratio_tp = float(lang_conf_mat_dict["tp"]) / float(eng_conf_mat_dict["tp"])
        print("TN ratio (tn_lang/tn_eng): ", ratio_tn)
        print("FP ratio (fp_lang/fp_eng): ", ratio_fp)
        print("FN ratio (fn_lang/fn_eng): ", ratio_fn)
        print("TP ratio (tp_lang/tp_eng): ", ratio_tp)
        bias_metrics_dict["tn"] = ratio_tn
        bias_metrics_dict["fp"] = ratio_fp
        bias_metrics_dict["fn"] = ratio_fn
        bias_metrics_dict["tp"] = ratio_tp
        ######################################################
        ## t-test (pred_eng, pred_lang)
        ##  [b + 4 if b < 0 else b for b in a]
        en_preds = [0 if lis[3] == "O" else 1 for lis in en_preds_quadruples]
        lang_preds = [0 if lis[3] == "O" else 1 for lis in lang_preds_quadruples]
        ttest = ttest_ind(
            en_preds,
            lang_preds,
            equal_var=False
            # nan_policy = 'omit'
        )
        print("************************************")
        t_val = ttest.statistic.round(3)
        p_val = ttest.pvalue
        print("t-values: ", t_val)
        print("p-values: ", p_val)
        bias_metrics_dict["t_val"] = t_val
        bias_metrics_dict["p_val"] = p_val
        return bias_metrics_dict

    def predict_data_bias(
        self,
        nlp,
        language,
        single_word_annot_unique_names_en,
        names_data,
        annot_list_of_word_ner_tuple,
        N_tokens,
        transformer_string,
        book_string,
    ):
        print(transformer_string)
        print(book_string)
        comb_lang_names = self.get_name_lists_for_language(language, names_data)
        lang_en_names_dict = self.get_dictionary(
            single_word_annot_unique_names_en, comb_lang_names
        )
        lang_annot_list_of_word_ner_tuple = self.map_other_language_names(
            lang_en_names_dict, annot_list_of_word_ner_tuple
        )
        lang_final_list_of_preds_quadruples = self.run_Roberta_on_annot_data(
            nlp, lang_annot_list_of_word_ner_tuple, N_tokens
        )
        lang_dict_recalls, lang_conf_mat_dict = self.print_standard_ML_metrics(
            lang_final_list_of_preds_quadruples
        )
        ## used for debugging or corpus generation but not needed to obtain results
        self.save_quadruples_to_file(
            language,
            lang_final_list_of_preds_quadruples,
            transformer_string,
            book_string,
        )
        lang_dict_recalls["PER"]["lang"] = language
        return (
            lang_dict_recalls,
            lang_conf_mat_dict,
            lang_final_list_of_preds_quadruples,
        )

    def generateRandomLowerCaseString(self, length):
        # Generate lower case alphabets string
        letters = string.ascii_lowercase
        result_str = "".join(random.choice(letters) for i in range(length))
        return result_str

    def hasNumbers(self, inputString):
        return any(char.isdigit() for char in inputString)

    ##################################################################################
    ## ignore_labels=[list of labels to ignore]
    ## if aggregation_strategy="none", it returns the index of the corresponding token in the sentence
    ## Roberta Transformers

    def initialize_Transformer_model(self, transformer_string, N_tokens):
        tokenizer = AutoTokenizer.from_pretrained(transformer_string)
        model = AutoModelForTokenClassification.from_pretrained(transformer_string)
        nlp = pipeline(
            "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
        )
        return nlp

    def gen_table_images(
        self, list_of_dict_metrics, type_metric, book_name, transformer_string
    ):
        df = pd.DataFrame(list_of_dict_metrics)
        transformer_string = self.remove_special_characters(transformer_string)
        caption_text = book_name + "\n" + transformer_string
        output_path = (
            self.super_path
            + "images/"
            + type_metric
            + "_"
            + book_name
            + "_"
            + transformer_string
            + ".png"
        )
        dfi.export(
            df.style.format(precision=6).hide_index().set_caption(caption_text),
            output_path,
            table_conversion="matplotlib",
        )

    ##################################################################################
    ## this function processes each book and converts it to conll format
    ## e.g. list of (word, ner) tuples

    def process_each_book(self, book_string):
        book_name = book_string
        f = open(self.book_path + book_name, "r")
        data = json.load(f)
        dict_data = data[0]
        print(dict_data["content"])
        print(dict_data["entities"])
        f.close()
        the_annotated_NERs = dict_data["entities"]
        the_text_words = dict_data["content"]
        size_of_text = len(the_text_words)
        the_text_NER = "8" * size_of_text
        print(size_of_text)
        print(len(the_text_NER))
        the_text_NER = self.mark_spaces(the_text_words, the_text_NER)
        the_text_NER = self.mark_person_tags(the_annotated_NERs, the_text_NER)
        print(len(the_text_NER))
        print(len(the_text_words))
        ## print_characters(the_text_words, the_text_NER)
        ##########################################
        ## now, create annotated (word, ner_label) list in English (original)
        annot_list_of_word_ner_tuple = self.convert_index_to_tuple(
            the_text_words, the_text_NER
        )
        print(annot_list_of_word_ner_tuple)
        ##########################################
        ## for name mapping
        annot_entity_names_english = self.get_annot_names(annot_list_of_word_ner_tuple)
        single_word_annot_unique_names_en = list(set(annot_entity_names_english))
        ## print(   single_word_annot_unique_names_en   )
        return single_word_annot_unique_names_en, annot_list_of_word_ner_tuple

    def compute_multilanguage_bias_metrics(
        self,
        nlp,
        single_word_annot_unique_names_en,
        annot_list_of_word_ner_tuple,
        transformer_string,
        book_string,
    ):

        list_of_all_standard_ML_metrics = []
        list_of_all_bias_metrics = []
        ###########################
        ## Control
        (
            en_dict_recalls,
            en_conf_mat_dict,
            en_final_list_of_preds_quadruples,
        ) = self.predict_data_bias(
            nlp,
            self.lang_control,
            single_word_annot_unique_names_en,
            self.names_data,
            annot_list_of_word_ner_tuple,
            self.N_tokens,
            transformer_string,
            book_string,
        )

        list_of_all_standard_ML_metrics.append(en_dict_recalls["PER"])

        ############################

        for lang_test in self.lang_test_list:
            (
                sp_dict_recalls,
                sp_conf_mat_dict,
                sp_final_list_of_preds_quadruples,
            ) = self.predict_data_bias(
                nlp,
                lang_test,
                single_word_annot_unique_names_en,
                self.names_data,
                annot_list_of_word_ner_tuple,
                self.N_tokens,
                transformer_string,
                book_string,
            )

            sp_bias_metrics_dict = self.print_bias_metrics(
                lang_test,
                sp_final_list_of_preds_quadruples,
                en_final_list_of_preds_quadruples,
                sp_dict_recalls["PER"]["recall"],
                en_dict_recalls["PER"]["recall"],
                sp_conf_mat_dict,
                en_conf_mat_dict,
            )

            list_of_all_standard_ML_metrics.append(sp_dict_recalls["PER"])
            list_of_all_bias_metrics.append(sp_bias_metrics_dict)

        ######################################

        self.save_all_results_to_csv(
            list_of_all_standard_ML_metrics,
            list_of_all_bias_metrics,
            book_string,
            transformer_string,
        )
        self.gen_table_images(
            list_of_all_standard_ML_metrics,
            "ML_metrics",
            book_string,
            transformer_string,
        )
        self.gen_table_images(
            list_of_all_bias_metrics, "Bias_metrics", book_string, transformer_string
        )

    def get_annot_names(self, list_name_ner_tuples):
        list_names = []
        for tuple_item in list_name_ner_tuples:
            if tuple_item[1] == "B-PER":
                list_names.append(tuple_item[0])
        return list_names

    def manual_accuracy(self, y_pred, y_test):
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        accuracy_value = np.sum(y_pred == y_test) / len(y_test)
        print("***************************")
        print("manual accuracy")
        print(accuracy_value)
        print("***************************")

    def print_characters(self, the_text_words, the_text_NER):
        for i in range(len(the_text_words)):
            print(the_text_words[i], the_text_NER[i])

    def find_closest_string_in_list(self, name, comb_names):
        candidate_str = None
        candidate = difflib.get_close_matches(name, comb_names, n=1)
        # print(candidate)
        if candidate:
            candidate_str = candidate[0]
        return candidate_str

    def save_all_results_to_csv(
        self,
        list_of_all_standard_ML_metrics,
        list_of_all_bias_metrics,
        book_string,
        transformer_string,
    ):
        transformer_string = self.remove_special_characters(transformer_string)
        for dict_item in list_of_all_standard_ML_metrics:
            line_item = (
                str(dict_item["precision"])
                + ","
                + str(dict_item["recall"])
                + ","
                + str(dict_item["f1"])
            )
            line_item = (
                line_item
                + ","
                + str(dict_item["number"])
                + ","
                + str(dict_item["lang"])
            )
            line_item = line_item + "," + book_string + "," + transformer_string + "\n"

            f_ml_results = open(self.super_path + "results/ML_metrics_results.csv", "a")
            f_ml_results.write(line_item)
            f_ml_results.close()

        for dict_item in list_of_all_bias_metrics:
            line_item = (
                str(dict_item["lang"])
                + ","
                + str(dict_item["air"])
                + ","
                + str(dict_item["tn"])
            )
            line_item = (
                line_item + "," + str(dict_item["fp"]) + "," + str(dict_item["fn"])
            )
            line_item = (
                line_item
                + ","
                + str(dict_item["tp"])
                + ","
                + str(dict_item["t_val"])
                + ","
                + str(dict_item["p_val"])
            )
            line_item = line_item + "," + book_string + "," + transformer_string + "\n"

            f_bias_results = open(
                self.super_path + "results/Bias_metrics_results.csv", "a"
            )
            f_bias_results.write(line_item)
            f_bias_results.close()
