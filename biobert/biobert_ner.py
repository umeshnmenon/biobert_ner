#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Umesh.Menon

BIOBER NER Inference

"""
import os, time, sys
libs_dir = os.path.dirname(os.path.abspath(__file__))
if libs_dir not in sys.path:
    sys.path.append(libs_dir)
import uuid
import tempfile
import tensorflow as tf
import spacy
from biobert import modeling
from biobert import tokenization
from biobert import run_ner
from biobert.run_ner import CustomNerProcessor, model_fn_builder, filed_based_convert_examples_to_features, \
                    file_based_input_fn_builder

# umesh added for testing
from timerr import Timer
from utils import get_full_path, get_fq_path, construct_fq_path, mkpath2
from log import *
setup_logger()

# TODO: Load this from a config file
#BERT_DIR = libs_dir + "/models/biobert_v1.0_pubmed_pmc/"
#FINE_TUNED_DIR = libs_dir + "/models/fine_tuned_ner"
#OUTPUT_DIR = libs_dir + "/tmp"
BERT_DIR = get_fq_path(libs_dir, "models", "biobert_v1.0_pubmed_pmc/")
FINE_TUNED_DIR = get_fq_path(libs_dir, "models", "fine_tuned_ner")
OUTPUT_DIR = get_fq_path(libs_dir, "tmp")
FINE_TUNED_MODEL_CKPT = 'model.ckpt-5143'
SPACY_MODEL = "en_core_web_sm"


# check if the tmp directory exists or not, if not, create one
if not mkpath2(OUTPUT_DIR):
    tf.logging.info("{} directory doesn't exist. Error occurred while creating the directory. Setting "
                    "output directory to {}.".format(OUTPUT_DIR, libs_dir))
    OUTPUT_DIR = libs_dir

# This is a WA to use flags from here:
flags = tf.flags

if 'f' not in tf.flags.FLAGS:
    tf.app.flags.DEFINE_string('f', '', 'kernel')

FLAGS = flags.FLAGS

if FLAGS.output_dir is None:
    FLAGS.output_dir = OUTPUT_DIR

class BioBertNER:
    """
    A minimalistic class the loads a Finetuned Biobert TensorFlow model to do NER
    """
    def __init__(self, model_dir=""):
        self.model_dir = model_dir # not used. Hard coded for now.
        self.load_model()
        self.nlp = spacy.load(SPACY_MODEL)

    @Timer(logging.getLogger())
    def load_model(self):
        """
        Loads a TF model from the given model directory
        :param model_dir:
        :return:
        """
        # The config json file corresponding to the pre-trained BERT model.
        # This specifies the model architecture.
        bert_config_file = os.path.join(BERT_DIR, 'bert_config.json')

        # The vocabulary file that the BERT model was trained on.
        vocab_file = os.path.join(BERT_DIR, 'vocab.txt')

        # initial checkpoint
        init_checkpoint = os.path.join(FINE_TUNED_DIR, FINE_TUNED_MODEL_CKPT)

        # create the input file from the given text
        tf.logging.set_verbosity(tf.logging.ERROR)

        # Whether to lower case the input text.
        # Should be True for uncased models and False for cased models.
        # The BioBERT available in NGC is uncased
        do_lower_case = True

        # Total batch size for predictions
        self.predict_batch_size = 1
        self.params = dict([('batch_size', self.predict_batch_size)])

        # The maximum total input sequence length after WordPiece tokenization.
        # Sequences longer than this will be truncated, and sequences shorter than this will be padded.
        self.max_seq_length = 128

        # Validate the casing config consistency with the checkpoint name.
        tokenization.validate_case_matches_checkpoint(do_lower_case, init_checkpoint)

        # Create the tokenizer.
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

        # Load the configuration from file
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)

        # Create the processor
        self.processor = CustomNerProcessor()

        # Get labels in the index order that was used during training
        self.label_list = self.processor.get_labels()

        # Reverse index the labels. This will be used later when evaluating predictions.
        self.id2label = {}
        for (i, label) in enumerate(self.label_list, 1):
            self.id2label[i] = label

        # Get run config
        # not using a TPU config as we are doing only inference
        #config = tf.ConfigProto(log_device_placement=True)
        #run_config = tf.estimator.RunConfig(
        #    model_dir=None, # or init_checkpoint
        #    session_config=config,
        #    save_checkpoints_steps=1000,
        #    keep_checkpoint_max=1)
        tpu_cluster_resolver = None
        if FLAGS.use_tpu and FLAGS.tpu_name:
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=FLAGS.master,
            model_dir=FINE_TUNED_DIR,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=FLAGS.iterations_per_loop,
                num_shards=FLAGS.num_tpu_cores,
                per_host_input_for_training=is_per_host))

        # Use model function builder to create the model function
        model_fn = model_fn_builder(
            bert_config=bert_config,
            num_labels=len(self.label_list) + 1,
            init_checkpoint=init_checkpoint)

        # Create the estimator
        # not using a TPU estimator as we are doing only inference
        #self.estimator = tf.estimator.Estimator(
        #    model_fn=model_fn,
        #    config=run_config,
        #    params=self.params)
        self.estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            predict_batch_size=self.predict_batch_size)

    @Timer(logging.getLogger())
    def create_bert_input_file(self, text):
        """
        Creates a temporary file using the input text
        :param text:
        :return:
        """
        # Convert the text into the IOB tags format seen during training, using dummy placeholder labels
        text = text.strip()
        doc = self.nlp(text)
        #temp = tempfile.NamedTemporaryFile(suffix="_input", prefix="tmp_", dir=OUTPUT_DIR, mode='w', delete=False)
        # below open command will work on Linux but not on windows. The reason is NamedTemporaryFile actually opens the
        # file, so windows doesn't allow you to open it for the second time
        # changing the temporary file way as it is not working the as expected in windows
        #with open(temp.name, 'w') as wf:
        #for word in doc:
        #    if word.text is '\n':
        #        continue
        #    temp.write(word.text + '\tO\n')
        #temp.write('\n')  # Indicate end of text
        #return temp
        #input_file_name = OUTPUT_DIR + "/" + "tmp_" + str(uuid.uuid4().hex) + "_input"
        input_file_name = get_fq_path(OUTPUT_DIR, "tmp_" + str(uuid.uuid4().hex) + "_input")
        with open(input_file_name, 'w') as wf:
            for word in doc:
                if word.text is '\n':
                    continue
                wf.write(word.text + '\tO\n')
            wf.write('\n')  # Indicate end of text
        return input_file_name

    @Timer(logging.getLogger())
    def predict(self, text):
        """
        Returns the named entities for the given text
        :param text:
        :return:
        """
        # Load the input data using the Custom NER processor
        #predict_examples = self.processor.get_pred_examples(text, self.nlp)
        # the above direct passing of text is not working as expected. REVISIT
        #temp_input = self.create_bert_input_file(text)
        #predict_examples = self.processor.get_test_examples(temp_input.name)
        #try:
        #    temp_input.close()
        #    os.unlink(temp_input.name)
        #except:
        #    pass
        temp_input = self.create_bert_input_file(text)
        predict_examples = self.processor.get_test_examples(temp_input)
        # Convert to tf_records and save it
        # create a temporary file
        # changing the temporary file way as it is not working the as expected in windows
        #temp = tempfile.NamedTemporaryFile(suffix="_tf_record", prefix="predict_", dir=OUTPUT_DIR, delete=False)
        #predict_file = os.path.join(OUTPUT_DIR, temp.name)
        #predict_file = OUTPUT_DIR + "/" + "predict_" + str(uuid.uuid4().hex) + "_tf_record"
        predict_file = construct_fq_path(OUTPUT_DIR, "predict_" + str(uuid.uuid4().hex) + "_tf_record")
        filed_based_convert_examples_to_features(predict_examples, self.label_list,
                                                 self.max_seq_length, self.tokenizer,
                                                 predict_file)

        # Run prediction on this tf_record file
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            #batch_size=self.predict_batch_size,
            seq_length=self.max_seq_length,
            is_training=False,
            drop_remainder=False)


        pred_start_time = time.time()

        predictions = self.estimator.predict(input_fn=predict_input_fn)
        predictions = list(predictions)
        if len(predictions) > 0:
            predictions = [predictions[0]["prediction"]]

        pred_time_elapsed = time.time() - pred_start_time

        tf.logging.info("Total Inference Time = %0.2f", pred_time_elapsed)
        # Convert the predictions to the Named Entities format required by displaCy and visualize
        ners = self.get_ners(predict_examples, predictions, self.id2label)

        #try:
        #    temp.close()
        #    os.unlink(temp.name)
        #except:
        #    pass
        try:
            os.remove(temp_input)
            os.remove(predict_file)
        except:
            pass
        return ners

    @Timer(logging.getLogger())
    def get_ners(self, predict_examples, predictions, id2label):
        processed_text = ''
        entities = []
        current_pos = 0
        start_pos = 0
        end_pos = 0
        end_detected = False
        prev_label = ''

        for predict_line, pred_ids in zip(predict_examples, predictions):
            words = str(predict_line.text).split(' ')
            labels = str(predict_line.label).split(' ')
            # get from CLS to SEP
            pred_labels = []
            for id in pred_ids:
                if id == 0:
                    continue
                curr_label = id2label[id]
                if curr_label == '[CLS]':
                    continue
                elif curr_label == '[SEP]':
                    break
                elif curr_label == 'X':
                    continue
                pred_labels.append(curr_label)

            for tok, label, pred_label in zip(words, labels, pred_labels):
                if pred_label is 'B':
                    start_pos = current_pos
                elif pred_label is 'I' and prev_label is not 'B' and prev_label is not 'I':
                    start_pos = current_pos
                elif pred_label is 'O' and (prev_label is 'B' or prev_label is 'I'):
                    end_pos = current_pos
                    end_detected = True

                if end_detected:
                    entities.append({'start': start_pos, 'end': end_pos, 'label': 'ENTITY'})
                    start_pos = 0
                    end_pos = 0
                    end_detected = False

                processed_text = processed_text + tok + ' '
                current_pos = current_pos + len(tok) + 1
                prev_label = pred_label

        # Handle entity at the very end
        if start_pos > 0 and end_detected is False:
            entities.append({'start': start_pos, 'end': current_pos, 'label': 'ENTITY'})

        ners = [{"text": processed_text,
                 "ents": entities,
                 "title": None}]
        ents = []
        for ent in entities:
            lbl = ent["label"]
            ne = processed_text[ent["start"]:ent["end"]].strip()
            ents.append({"text": ne, "label_": lbl})

        return ners, ents
