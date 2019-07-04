import re
import tensorflow as tf


def _space_around_punctuation(text):
    return re.sub('(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )', r' ', text)


def clean_text(text):
    text = str(text)
    text = text.strip().replace('\n', ' ')
    text = text.replace('\t', ' ')
    # text = text.strip().replace('.', ' ')
    # text = text.replace('#_new_utterance_#', ' \t ')
    text = _space_around_punctuation(text)
    text = ' '.join(text.split())
    return text


def obtain_samples(path):
    """Return list of samples from given path."""
    samples = []
    for i, record in enumerate(tf.python_io.tf_record_iterator(path)):
        example = tf.train.Example()
        example.ParseFromString(record)
        samples.append(example)
    return samples


def get_additional_context(sample):
    context = []
    i = 0
    while 1:
        try:
            feature = 'context/{}'.format(i)
            feature_val = get_feature_value(sample, feature)
            context.append(clean_text(feature_val))
            i += 1
        except IndexError:
            break
    return context


def get_feature_value(sample, feature):
    return sample.features.feature[feature].bytes_list.value[0].decode("utf-8")
