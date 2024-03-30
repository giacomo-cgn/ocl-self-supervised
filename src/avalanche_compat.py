import torch

from avalanche.benchmarks import GenericCLScenario, ClassificationExperience, ClassificationStream

from itertools import tee

# IMPLEMENTED FUNCTIONS TAKEN AVALANCHE TO ENSURE COMPATIBILITY WITH OLDER VERSIONS OF AVALANCHE


def _lazy_train_val_split(
    split_strategy,
    experiences):
    """
    Creates a generator operating around the split strategy and the
    experiences stream.

    :param split_strategy: The strategy used to split each experience in train
        and validation datasets.
    :return: A generator returning a 2 elements tuple (the train and validation
        datasets).
    """

    for new_experience in experiences:
        yield split_strategy(new_experience)


def _gen_split(
    split_generator):
    """
    Internal utility function to split the train-validation generator
    into two distinct generators (one for the train stream and another one
    for the valid stream).

    :param split_generator: The lazy stream generator returning tuples of train
        and valid datasets.
    :return: Two generators (one for the train, one for the valuid).
    """

    # For more info: https://stackoverflow.com/a/28030261
    gen_a, gen_b = tee(split_generator, 2)
    return (a for a, b in gen_a), (b for a, b in gen_b)



def class_balanced_split_strategy(validation_size, experience):
    """Class-balanced train/validation splits.

    This splitting strategy splits `experience` into two experiences
    (train and validation) of size `validation_size` using a class-balanced
    split. Sample of each class are chosen randomly.

    You can use this split strategy to split a benchmark with::

        validation_size = 0.2
        foo = lambda exp: class_balanced_split_strategy(validation_size, exp)
        bm = benchmark_with_validation_stream(bm, custom_split_strategy=foo)

    :param validation_size: The percentage of samples to allocate to the
        validation experience as a float between 0 and 1.
    :param experience: The experience to split.
    :return: A tuple containing 2 elements: the new training and validation
        datasets.
    """
    if not isinstance(validation_size, float):
        raise ValueError("validation_size must be an integer")
    if not 0.0 <= validation_size <= 1.0:
        raise ValueError("validation_size must be a float in [0, 1].")

    exp_dataset = experience.dataset
    if validation_size > len(exp_dataset):
        raise ValueError(
            f"Can't create the validation experience: not enough "
            f"instances. Required {validation_size}, got only"
            f"{len(exp_dataset)}"
        )

    exp_indices = list(range(len(exp_dataset)))
    exp_classes = experience.classes_in_this_experience

    # shuffle exp_indices
    exp_indices = torch.as_tensor(exp_indices)[torch.randperm(len(exp_indices))]
    # shuffle the targets as well
    exp_targets = torch.as_tensor(experience.dataset.targets)[exp_indices]

    train_exp_indices = []
    valid_exp_indices = []
    for cid in exp_classes:  # split indices for each class separately.
        c_indices = exp_indices[exp_targets == cid]
        valid_n_instances = int(validation_size * len(c_indices))
        valid_exp_indices.extend(c_indices[:valid_n_instances])
        train_exp_indices.extend(c_indices[valid_n_instances:])

    result_train_dataset = exp_dataset.subset(train_exp_indices)
    result_valid_dataset = exp_dataset.subset(valid_exp_indices)
    return result_train_dataset, result_valid_dataset




from avalanche.benchmarks.scenarios.classification_scenario import (
    TStreamsUserDict,
    StreamUserDef,
)

def benchmark_with_validation_stream(
    benchmark_instance: GenericCLScenario,
    validation_size = 0.5,
    shuffle: bool = False,
    input_stream: str = "train",
    output_stream: str = "valid",
    custom_split_strategy = None,
    *,
    experience_factory = None,
    lazy_splitting: bool = None,
):
    """
    Helper that can be used to obtain a benchmark with a validation stream.

    This generator accepts an existing benchmark instance and returns a version
    of it in which a validation stream has been added.

    In its base form this generator will split train experiences to extract
    validation experiences of a fixed (by number of instances or relative
    size), configurable, size. The split can be also performed on other
    streams if needed and the name of the resulting validation stream can
    be configured too.

    Each validation experience will be extracted directly from a single training
    experience. Patterns selected for the validation experience will be removed
    from the training one.

    If shuffle is True, the validation stream will be created randomly.
    Beware that no kind of class balancing is done.

    The `custom_split_strategy` parameter can be used if a more specific
    splitting is required.

    Please note that the resulting experiences will have a task labels field
    equal to the one of the originating experience.

    Experience splitting can be executed in a lazy way. This behavior can be
    controlled using the `lazy_splitting` parameter. By default, experiences
    are split in a lazy way only when the input stream is lazily generated.

    :param benchmark_instance: The benchmark to split.
    :param validation_size: The size of the validation experience, as an int
        or a float between 0 and 1. Ignored if `custom_split_strategy` is used.
    :param shuffle: If True, patterns will be allocated to the validation
        stream randomly. This will use the default PyTorch random number
        generator at its current state. Defaults to False. Ignored if
        `custom_split_strategy` is used. If False, the first instances will be
        allocated to the training  dataset by leaving the last ones to the
        validation dataset.
    :param input_stream: The name of the input stream. Defaults to 'train'.
    :param output_stream: The name of the output stream. Defaults to 'valid'.
    :param custom_split_strategy: A function that implements a custom splitting
        strategy. The function must accept an experience and return a tuple
        containing the new train and validation dataset. Defaults to None,
        which means that the standard splitting strategy will be used (which
        creates experiences according to `validation_size` and `shuffle`).
        A good starting to understand the mechanism is to look at the
        implementation of the standard splitting function
        :func:`random_validation_split_strategy`.
    :param experience_factory: The experience factory. Defaults to
        :class:`GenericExperience`.
    :param lazy_splitting: If True, the stream will be split in a lazy way.
        If False, the stream will be split immediately. Defaults to None, which
        means that the stream will be split in a lazy or non-lazy way depending
        on the laziness of the `input_stream`.
    :return: A benchmark instance in which the validation stream has been added.
    """

    split_strategy = custom_split_strategy
    if split_strategy is None:
        raise Exception('Include split strategy')

    stream_definitions: TStreamsUserDict = dict(
        benchmark_instance.stream_definitions
    )
    streams = benchmark_instance.streams

    if input_stream not in streams:
        raise ValueError(
            f"Stream {input_stream} could not be found in the "
            f"benchmark instance"
        )

    if output_stream in streams:
        raise ValueError(
            f"Stream {output_stream} already exists in the "
            f"benchmark instance"
        )

    stream = streams[input_stream]

    split_lazily = lazy_splitting
    if split_lazily is None:
        split_lazily = stream_definitions[input_stream].is_lazy

    exps_tasks_labels = list(stream_definitions[input_stream].exps_task_labels)

    if not split_lazily:
        # Classic static splitting
        train_exps_source = []
        valid_exps_source = []

        exp: ClassificationExperience
        for exp in stream:
            train_exp, valid_exp = split_strategy(exp)
            train_exps_source.append(train_exp)
            valid_exps_source.append(valid_exp)
    else:
        # Lazy splitting (based on a generator)
        split_generator = _lazy_train_val_split(split_strategy, stream)
        train_exps_gen, valid_exps_gen = _gen_split(split_generator)
        train_exps_source = (train_exps_gen, len(stream))
        valid_exps_source = (valid_exps_gen, len(stream))

    train_stream_def = StreamUserDef(
        train_exps_source,
        exps_tasks_labels,
        stream_definitions[input_stream].origin_dataset,
        split_lazily,
    )

    valid_stream_def = StreamUserDef(
        valid_exps_source,
        exps_tasks_labels,
        stream_definitions[input_stream].origin_dataset,
        split_lazily,
    )

    stream_definitions[input_stream] = train_stream_def
    stream_definitions[output_stream] = valid_stream_def

    complete_test_set_only = benchmark_instance.complete_test_set_only

    return GenericCLScenario(
        stream_definitions=stream_definitions,
        complete_test_set_only=complete_test_set_only,
        experience_factory=experience_factory,
    )
       