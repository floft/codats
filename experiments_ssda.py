#!/usr/bin/env python3
"""
Generates the list of which single-source adaptation problems to perform

For each dataset, generate 10 random source-target pairs (excluding the source
as the target, otherwise it's not domain adaptation)

Note: 3 runs for each, but that's in the .srun scripts not here
(for mean +/- stdev)

Usage: ./experiments_ssda.py > experiments_ssda.txt
"""
import random
import itertools

import datasets.datasets as datasets

from experiments_msda import natural_keys


def generate_single_source(dataset_name, users, max_number=5):
    # Take random set of the possible combinations
    combinations = list(itertools.combinations(users, 2))
    random.shuffle(combinations)
    combinations = combinations[:max_number]

    pairs = []

    for source_user, target_user in combinations:
        assert source_user != target_user
        pairs.append((dataset_name, str(source_user), str(target_user)))

    return pairs


if __name__ == "__main__":
    # Sources-target pairs for training
    pairs = []
    uids = []

    for name in datasets.list_datasets():
        # Tune on "watch_noother" not "watch"
        if name == "watch":
            continue

        users = datasets.get_dataset_users(name)

        # Since sources-target aren't stored in filename anymore (too long), we
        # would run into folder name conflicts if we didn't append a unique ID
        # to each sources-target pair
        uid = 0

        # Make this repeatable
        random.seed(42)

        # Allows extra max_users for some datasets without changin uid's
        #
        # TODO get rid of all this confusing code once we decide what number
        # to set max_users to. If we don't need to change max_users, then
        # we can just increment uid's like before.
        bonus_uid = 0

        max_number = 10

        curr_pairs = generate_single_source(name, users, max_number=max_number)

        for i, (dataset_name, source_users, target_user) in enumerate(curr_pairs):
            # We want to allow increasing the number of max_users for
            # wisdm_at and watch without changing the uid's of the 0-4
            # targets for backwards compatibility (otherwise we have to move
            # all the models around...)
            set_of_five = i // 5

            # before we had 0-4 (or 1-5), so do as before
            if max_number == 5 or set_of_five == 0:
                uids.append(uid)
                uid += 1
            else:
                uids.append(str(uid)+"_"+str(bonus_uid))
                bonus_uid += 1

        pairs += curr_pairs

    # Check that these make sense
    print("List of adaptations we'll perform:")
    for i, (dataset_name, source, target) in enumerate(pairs):
        print("    ", dataset_name, source, "to", target, "uid", uids[i])
    print()

    #
    # kamiak_{train,eval}_ssda.srun
    #
    print("For kamiak_{train,eval}_ssda.srun:")
    dataset_names = []
    print_uids = []
    sources = []
    targets = []
    dataset_target_pairs = {}  # for upper bounds
    for i, (dataset_name, source, target) in enumerate(pairs):
        dataset_names.append("\""+dataset_name+"\"")
        print_uids.append(str(uids[i]))
        sources.append("\""+source+"\"")
        targets.append("\""+target+"\"")

        # for upper bounds
        pair_name = ("\""+dataset_name+"\"", "\""+target+"\"")
        full_pair = ("\""+dataset_name+"\"", str(uids[i]), "\""+target+"\"")
        if pair_name not in dataset_target_pairs:
            dataset_target_pairs[pair_name] = full_pair

    print("# number of adaptation problems =", len(sources))
    print("uids=(", " ".join(print_uids), ")", sep="")
    print("datasets=(", " ".join(dataset_names), ")", sep="")
    print("sources=(", " ".join(sources), ")", sep="")
    print("targets=(", " ".join(targets), ")", sep="")
    print()

    #
    # kamiak_{train,eval}_ssda_upper.srun
    #
    print("For kamiak_{train,eval}_ssda_upper.srun:")
    targets_unique = list(set(dataset_target_pairs.values()))
    targets_unique.sort(key=natural_keys)
    sources_blank = ["\"\""]*len(targets_unique)

    targets_unique_uids = []
    targets_unique_dataset = []
    targets_unique_target = []

    for dataset_name, uid, target in targets_unique:
        # Uses first uid from dataset_name-target
        targets_unique_uids.append(uid)
        targets_unique_dataset.append(dataset_name)
        targets_unique_target.append(target)

    print("# number of adaptation problems =", len(targets_unique))
    print("uids=(", " ".join(["u"+str(x) for x in targets_unique_uids]), ")", sep="")
    print("datasets=(", " ".join(targets_unique_dataset), ")", sep="")
    print("sources=(", " ".join(sources_blank), ")", sep="")
    print("targets=(", " ".join(targets_unique_target), ")", sep="")
    print()
