#!/usr/bin/env python3
"""
Generates the list of which multi-source adaptation problems to perform

For each dataset, for each target user, pick n random source users (excluding
the target user) 3 different times (so we can get mean +/- stdev).

Usage: ./experiments_msda.py > experiments_msda.txt
"""
import re
import random

import datasets.datasets as datasets


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
    #
    # Take random set though, since IDs aren't necessarily randomized.
    # Note: not using random.shuffle() since that shuffles in-place
    shuffled_users = random.sample(users, len(users))
    possible_target_users = shuffled_users[:max_users]

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

    for name in datasets.list_datasets():
        users = datasets.get_dataset_users(name)

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

            max_users = 10

            curr_pairs = generate_multi_source(name, users, n,
                max_users=max_users)

            for i, (dataset_name, source_users, target_user) in enumerate(curr_pairs):
                # We want to allow increasing the number of max_users for
                # wisdm_at and watch without changing the uid's of the 0-4
                # targets for backwards compatibility (otherwise we have to move
                # all the models around...)
                set_of_five = i // 5

                # before we had 0-4 (or 1-5), so do as before
                if max_users == 5 or set_of_five == 0:
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
    # kamiak_{train,eval}_msda.srun
    #
    print("For kamiak_{train,eval}_msda.srun:")
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
    # kamiak_{train,eval}_msda_upper.srun
    #
    print("For kamiak_{train,eval}_msda_upper.srun:")
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
