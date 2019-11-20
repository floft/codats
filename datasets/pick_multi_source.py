#!/usr/bin/env python3
"""
Generates the list of which multi-source adaptation problems to perform

For each dataset, for each target user, pick n random source users (excluding
the target user) 3 different times (so we can get mean +/- stdev).
"""
import re
import random

from datasets import dataset_users


# ./hyperparameters.py --selection=best_source
hyperparameters_source = {
    "ucihar": {
        "aflac_dg": "--batch_division=all --train_batch=256 --lr=0.001",
        "dann": "--batch_division=sources --train_batch=128 --lr=0.01",
        "dann_dg": "--batch_division=sources --train_batch=256 --lr=0.01",
        "dann_gs": "--batch_division=sources --train_batch=256 --lr=0.01",
        "dann_smooth": "--batch_division=sources --train_batch=256 --lr=0.001",
        "none": "--batch_division=all --train_batch=256 --lr=0.001",
        "sleep_dg": "--batch_division=sources --train_batch=256 --lr=0.01",
    },
    "ucihhar": {
        "aflac_dg": "--batch_division=sources --train_batch=256 --lr=0.001",
        "dann": "--batch_division=sources --train_batch=256 --lr=0.01",
        "dann_dg": "--batch_division=all --train_batch=256 --lr=0.01",
        "dann_gs": "--batch_division=sources --train_batch=256 --lr=0.01",
        "dann_smooth": "--batch_division=sources --train_batch=256 --lr=0.01",
        "none": "--batch_division=sources --train_batch=256 --lr=0.01",
        "sleep_dg": "--batch_division=sources --train_batch=256 --lr=0.01",
    },
    "uwave": {
        "aflac_dg": "--batch_division=all --train_batch=128 --lr=0.0001",
        "dann": "--batch_division=all --train_batch=64 --lr=0.01",
        "dann_dg": "--batch_division=sources --train_batch=64 --lr=0.01",
        "dann_gs": "--batch_division=sources --train_batch=64 --lr=0.01",
        "dann_smooth": "--batch_division=all --train_batch=64 --lr=0.0001",
        #"dann_smooth": "--batch_division=all --train_batch=256 --lr=0.0001",
        "none": "--batch_division=sources --train_batch=128 --lr=0.001",
        #"none": "--batch_division=all --train_batch=256 --lr=0.001",
        "sleep_dg": "--batch_division=sources --train_batch=256 --lr=0.01",
    },
    "wisdm_ar": {
        "aflac_dg": "--batch_division=sources --train_batch=64 --lr=0.01",
        "dann": "--batch_division=sources --train_batch=256 --lr=0.01",
        "dann_dg": "--batch_division=sources --train_batch=64 --lr=0.01",
        "dann_gs": "--batch_division=all --train_batch=128 --lr=0.01",
        "dann_smooth": "--batch_division=sources --train_batch=64 --lr=0.001",
        "none": "--batch_division=all --train_batch=256 --lr=0.01",
        "sleep_dg": "--batch_division=sources --train_batch=64 --lr=0.01",
    },
    "wisdm_at": {
        "aflac_dg": "--batch_division=all --train_batch=256 --lr=0.001",
        "dann": "--batch_division=sources --train_batch=256 --lr=0.01",
        "dann_dg": "--batch_division=all --train_batch=256 --lr=0.01",
        "dann_gs": "--batch_division=all --train_batch=256 --lr=0.01",
        "dann_smooth": "--batch_division=sources --train_batch=256 --lr=0.0001",
        "none": "--batch_division=all --train_batch=256 --lr=0.001",
        "sleep_dg": "--batch_division=all --train_batch=256 --lr=0.01",
    },
    "watch_noother": {
        "aflac_dg": "--batch_division=sources --train_batch=256 --lr=0.001",
        "dann": "--batch_division=sources --train_batch=256 --lr=0.01",
        "dann_dg": "--batch_division=sources --train_batch=256 --lr=0.01",
        "dann_gs": "--batch_division=all --train_batch=256 --lr=0.001",
        "dann_smooth": "--batch_division=sources --train_batch=256 --lr=0.001",
        "none": "--batch_division=all --train_batch=256 --lr=0.001",
        "sleep_dg": "--batch_division=all --train_batch=256 --lr=0.01",
    },
}

# ./hyperparameters.py --selection=best_target
hyperparameters_target = {
    "ucihar": {
        "aflac_dg": "--batch_division=all --train_batch=256 --lr=0.001",
        "dann": "--batch_division=sources --train_batch=256 --lr=0.01",
        "dann_dg": "--batch_division=all --train_batch=256 --lr=0.01",
        "dann_gs": "--batch_division=sources --train_batch=256 --lr=0.01",
        "dann_smooth": "--batch_division=sources --train_batch=256 --lr=0.001",
        "none": "--batch_division=all --train_batch=256 --lr=0.01",
        #"none": "--batch_division=sources --train_batch=256 --lr=0.001",
        "sleep_dg": "--batch_division=all --train_batch=256 --lr=0.01",
    },
    "ucihhar": {
        "aflac_dg": "--batch_division=all --train_batch=64 --lr=0.01",
        "dann": "--batch_division=all --train_batch=256 --lr=0.0001",
        "dann_dg": "--batch_division=sources --train_batch=64 --lr=0.0001",
        "dann_gs": "--batch_division=all --train_batch=256 --lr=0.001",
        "dann_smooth": "--batch_division=all --train_batch=256 --lr=0.0001",
        "none": "--batch_division=all --train_batch=128 --lr=0.001",
        "sleep_dg": "--batch_division=all --train_batch=64 --lr=0.0001",
    },
    "uwave": {
        "aflac_dg": "--batch_division=sources --train_batch=64 --lr=0.0001",
        "dann": "--batch_division=all --train_batch=64 --lr=0.0001",
        "dann_dg": "--batch_division=sources --train_batch=128 --lr=0.01",
        "dann_gs": "--batch_division=all --train_batch=256 --lr=0.0001",
        "dann_smooth": "--batch_division=all --train_batch=64 --lr=0.0001",
        #"dann_smooth": "--batch_division=all --train_batch=256 --lr=0.0001",
        "none": "--batch_division=all --train_batch=64 --lr=0.001",
        "sleep_dg": "--batch_division=all --train_batch=256 --lr=0.01",
    },
    "wisdm_ar": {
        "aflac_dg": "--batch_division=sources --train_batch=64 --lr=0.0001",
        "dann": "--batch_division=all --train_batch=64 --lr=0.01",
        "dann_dg": "--batch_division=all --train_batch=64 --lr=0.0001",
        "dann_gs": "--batch_division=all --train_batch=64 --lr=0.001",
        "dann_smooth": "--batch_division=all --train_batch=256 --lr=0.001",
        "none": "--batch_division=all --train_batch=64 --lr=0.0001",
        "sleep_dg": "--batch_division=sources --train_batch=128 --lr=0.0001",
    },
    "wisdm_at": {
        "aflac_dg": "--batch_division=all --train_batch=256 --lr=0.0001",
        "dann": "--batch_division=all --train_batch=256 --lr=0.0001",
        "dann_dg": "--batch_division=sources --train_batch=64 --lr=0.001",
        "dann_gs": "--batch_division=all --train_batch=64 --lr=0.0001",
        "dann_smooth": "--batch_division=all --train_batch=128 --lr=0.0001",
        "none": "--batch_division=sources --train_batch=64 --lr=0.01",
        "sleep_dg": "--batch_division=sources --train_batch=128 --lr=0.01",
    },
    "watch_noother": {
        "aflac_dg": "--batch_division=all --train_batch=64 --lr=0.001",
        "dann": "--batch_division=sources --train_batch=128 --lr=0.01",
        "dann_dg": "--batch_division=all --train_batch=64 --lr=0.001",
        "dann_gs": "--batch_division=all --train_batch=64 --lr=0.001",
        "dann_smooth": "--batch_division=all --train_batch=128 --lr=0.001",
        "none": "--batch_division=sources --train_batch=128 --lr=0.001",
        "sleep_dg": "--batch_division=sources --train_batch=128 --lr=0.01",
    },
}


def other_users(users, skip_user):
    """ From the list of users, throw out skip_user """
    new_users = []

    for user in users:
        if user != skip_user:
            new_users.append(user)

    return new_users


def generate_n_with_max(num_users, max_num):
    """ Generate [1,2,3,...,num_users] but max out at max_num and skip as close
    to evenly to get there. For example, if num_users=30 and max_num=5, we get:
    [1, 7, 13, 19, 25].
    """
    return list(range(1, num_users, num_users//max_num))[:max_num]


def generate_multi_source(dataset_name, users, n, repeat=3, max_users=5):
    # Shrink the number of target users since otherwise we have >4000 adaptation
    # problems. That will take too long and won't fit in the paper's table
    # anyway.
    possible_target_users = users[:max_users]

    # We'll generate multi-source options for each target user
    pairs = []

    for target_user in possible_target_users:
        already_used_target = {}

        # We want several random subsets of each so we can get mean +/- stdev
        for i in range(repeat):
            skip = False

            # Select random source domains excluding target, keep shuffling until
            # we find a source set we haven't already used. The point of "repeat"
            # is to get *different* subsets. If it's the same, then there's not
            # much point in re-running with the exact same data.
            j = 0
            while True:
                others = other_users(users, target_user)
                random.shuffle(others)
                assert n <= len(others), "cannot choose n larger than len(users)-1"
                source_users = others[:n]

                # Sort so if we ever use the same subset, we don't have to
                # regenerate the files. Also easier to read.
                source_users.sort()

                if tuple(source_users) not in already_used_target:
                    already_used_target[tuple(source_users)] = None
                    break
                elif j > 1000:
                    print("Warning: couldn't pick different set of sources",
                        "than previously used,",
                        "dataset:"+dataset_name+",",
                        "n:"+str(n)+",",
                        "user:"+str(target_user)+",",
                        "repeat:"+str(i))
                    skip = True
                    break
                j += 1

            # Skip if this "repeat" would be the same as a previous one
            if skip:
                continue

            source_users = ",".join([str(x) for x in source_users])
            pairs.append((dataset_name, source_users, str(target_user)))

    return pairs


def get_tuning_params():
    """ Parameters we vary during hyperparameter tuning
    batch_division - divide batch evenly among all domains or have same amount
        of source and target data
    lr - three values, 0.0001 is what we used in previous paper
    batch - three values, centered around what we used in previous paper
    """
    tuning_params = []

    for division in ["sources", "all"]:
        for lr in [0.01, 0.001, 0.0001]:
            for batch in [64, 128, 256]:
                params = {
                    "batch_division": division,
                    "train_batch": batch,
                    "lr": lr,
                }
                tuning_params.append(params)

    return tuning_params


def atof(text):
    """ https://stackoverflow.com/a/5967539 """
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    """
    https://stackoverflow.com/a/5967539
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    """
    text = text[0] + text[1]  # we actually are sorting tuples of strings
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]


if __name__ == "__main__":
    # Sources-target pairs for training
    pairs = []
    uids = []

    # Hyperparameter tuning
    tuning = []
    tuning_params = get_tuning_params()
    tuning_uid = 0

    # Note: "dataset_users" is set in datasets.py
    for name, users in dataset_users.items():
        # Tune on "watch_noother" not "watch"
        if name == "watch":
            continue

        # Since sources-target aren't stored in filename anymore (too long), we
        # would run into folder name conflicts if we didn't append a unique ID
        # to each sources-target pair
        uid = 0

        # For each value of n, from 1 (single-source domain adaptation) up to
        # the full number of users - 1 (since we have one for the target)
        options = generate_n_with_max(len(users), 5)

        for i, n in enumerate(options):
            # Make this repeatable even if we change which datasets, how many
            # n's we use, etc. Also nice since we end up using a subset of
            # n's source domains as (n-1)'s source domains. For example,
            # we get
            # (dataset_name, source_users, target_user) where each is a string
            # "sleep", "17", "0"
            # "sleep", "17,13", "0"
            # "sleep", "17,13,10", "0"
            # "sleep", "17,13,10,20", "0"
            random.seed(42)

            # Allows extra max_users for some datasets without changin uid's
            #
            # TODO get rid of all this confusing code once we decide what number
            # to set max_users to. If we don't need to change max_users, then
            # we can just increment uid's like before.
            bonus_uid = 0

            if name == "wisdm_at":
                max_users = 10  # Note: we only used 5 for tuning though
            else:
                max_users = 5

            curr_pairs = generate_multi_source(name, users, n,
                max_users=max_users)

            for dataset_name, source_users, target_user in curr_pairs:
                # We want to allow increasing the number of max_users for
                # wisdm_at and watch without changing the uid's of the 0-4
                # targets for backwards compatibility (otherwise we have to move
                # all the models around...)
                set_of_five = int(target_user) // 5

                # before we had 0-4 (or 1-5), so do as before
                if max_users == 5 or set_of_five == 0:
                    uids.append(uid)
                    uid += 1
                else:
                    uids.append(str(uid)+"_"+str(bonus_uid))
                    bonus_uid += 1

            # Save highest one for hyperparameter tuning
            if i == len(options)-1:
                for pair in curr_pairs:
                    for params in tuning_params:
                        tuning.append((tuning_uid, params, pair))
                        tuning_uid += 1

            pairs += curr_pairs

    # Check that these make sense
    print("List of adaptations we'll perform:")
    for i, (dataset_name, source, target) in enumerate(pairs):
        print("    ", dataset_name, source, "to", target, "uid", uids[i])
    print()

    #
    # kamiak_train_real_{source,target}.srun
    #
    # List of methods (excluding "upper", which is run separately)
    # We need to unwrap the methods dimension from the slurm array because we
    # have to specify different hyperparameters for each dataset-method pair.
    method_list = [
        "dann_smooth",
        "dann",
        "dann_dg",
        "sleep_dg",
        "aflac_dg",
        "dann_gs",
        "none"
    ]

    print("For kamiak_train_real_source.srun:")
    methods = []
    print_uids = []
    dataset_names = []
    sources = []
    targets = []
    other_params = []
    for method in method_list:
        for i, (dataset_name, source, target) in enumerate(pairs):
            if dataset_name not in hyperparameters_source:
                print("Warning: skipping dataset", dataset_name, "since no hyperparameters")
                continue

            # TODO remove
            if "watch" not in dataset_name:
                continue

            methods.append("\""+method+"\"")
            print_uids.append(str(uids[i]))
            dataset_names.append("\""+dataset_name+"\"")
            sources.append("\""+source+"\"")
            targets.append("\""+target+"\"")
            other_params.append(("\""+hyperparameters_source[dataset_name][method]+"\""))

    print("# number of adaptation problems =", len(sources))
    print("methods=(", " ".join(methods), ")", sep="")
    print("uids=(", " ".join(print_uids), ")", sep="")
    print("datasets=(", " ".join(dataset_names), ")", sep="")
    print("sources=(", " ".join(sources), ")", sep="")
    print("targets=(", " ".join(targets), ")", sep="")
    print("other_params=(", " ".join(other_params), ")", sep="")
    print()

    # Note: difference from above is different hyperparameters
    print("For kamiak_train_real_target.srun:")
    methods = []
    print_uids = []
    dataset_names = []
    sources = []
    targets = []
    other_params = []
    for method in method_list:
        for i, (dataset_name, source, target) in enumerate(pairs):
            if dataset_name not in hyperparameters_target:
                #print("Warning: skipping dataset", dataset_name, "since no hyperparameters")
                continue

            # TODO remove
            if "watch" not in dataset_name:
                continue

            methods.append("\""+method+"\"")
            print_uids.append(str(uids[i]))
            dataset_names.append("\""+dataset_name+"\"")
            sources.append("\""+source+"\"")
            targets.append("\""+target+"\"")
            other_params.append(("\""+hyperparameters_target[dataset_name][method]+"\""))

    print("# number of adaptation problems =", len(sources))
    print("methods=(", " ".join(methods), ")", sep="")
    print("uids=(", " ".join(print_uids), ")", sep="")
    print("datasets=(", " ".join(dataset_names), ")", sep="")
    print("sources=(", " ".join(sources), ")", sep="")
    print("targets=(", " ".join(targets), ")", sep="")
    print("other_params=(", " ".join(other_params), ")", sep="")
    print()

    #
    # kamiak_eval_real_{source,target}.srun (same as above, but don't need to
    # unwrap method and don't need other_params)
    #
    print("For kamiak_eval_real_{source,target}.srun:")
    dataset_names = []
    print_uids = []
    sources = []
    targets = []
    dataset_target_pairs = {}  # for upper bounds
    for i, (dataset_name, source, target) in enumerate(pairs):
        # If we didn't train them, then don't evaluate it either
        if dataset_name not in hyperparameters_source \
                and dataset_name not in hyperparameters_target:
            continue

        # TODO remove
        if "watch" not in dataset_name:
            continue

        dataset_names.append("\""+dataset_name+"\"")
        print_uids.append(str(uids[i]))
        sources.append("\""+source+"\"")
        targets.append("\""+target+"\"")

        # for upper bounds
        pair_name = ("\""+dataset_name+"\"", "\""+target+"\"")
        if pair_name not in dataset_target_pairs:
            dataset_target_pairs[pair_name] = str(uids[i])

    print("# number of adaptation problems =", len(sources))
    print("uids=(", " ".join(print_uids), ")", sep="")
    print("datasets=(", " ".join(dataset_names), ")", sep="")
    print("sources=(", " ".join(sources), ")", sep="")
    print("targets=(", " ".join(targets), ")", sep="")
    print()

    #
    # kamiak_{train,eval}_real_upper.srun
    #
    print("For kamiak_{train,eval}_real_upper.srun:")
    targets_unique = list(set(dataset_target_pairs.keys()))
    targets_unique.sort(key=natural_keys)
    sources_blank = ["\"\""]*len(targets_unique)

    uid = 0
    targets_unique_uids = []
    targets_unique_dataset = []
    targets_unique_target = []

    for dataset_name, target in targets_unique:
        # Uses first uid from dataset_name-target
        targets_unique_uids.append(dataset_target_pairs[(dataset_name, target)])
        uid += 1
        targets_unique_dataset.append(dataset_name)
        targets_unique_target.append(target)

    print("# number of adaptation problems =", len(targets_unique))
    print("uids=(", " ".join(["u"+str(x) for x in targets_unique_uids]), ")", sep="")
    print("datasets=(", " ".join(targets_unique_dataset), ")", sep="")
    print("sources=(", " ".join(sources_blank), ")", sep="")
    print("targets=(", " ".join(targets_unique_target), ")", sep="")
    print()

    #
    # kamiak_{train,eval}_tune.srun
    #
    print("For kamiak_{train,eval}_tune.srun (skip other_params in eval script though):")
    uids = []
    dataset_names = []
    sources = []
    targets = []
    other_params = []
    for tuning_uid, params, (dataset_name, source, target) in tuning:
        hyper_params = " ".join(["--"+k+"="+str(v) for k, v in params.items()])

        uids.append(tuning_uid)
        dataset_names.append("\""+dataset_name+"\"")
        sources.append("\""+source+"\"")
        targets.append("\""+target+"\"")
        other_params.append(("\""+hyper_params+"\""))

    print("# number of adaptation problems =", len(sources))
    print("uids=(", " ".join([str(x) for x in uids]), ")", sep="")
    print("datasets=(", " ".join(dataset_names), ")", sep="")
    print("sources=(", " ".join(sources), ")", sep="")
    print("targets=(", " ".join(targets), ")", sep="")
    print("other_params=(", " ".join(other_params), ")", sep="")
    print()
