import time
import numpy as np
import json
import sys

def time_to_features(timestamp, data, pos):
    dt_object = time.localtime(timestamp)
    # data = []
    data[pos] = (dt_object.tm_mday / 31.0)
    data[pos+1] = (dt_object.tm_sec / 60.0)
    data[pos+2] = (dt_object.tm_min / 59.0)
    data[pos+3] = (dt_object.tm_hour / 23.0)
    data[pos+4] = ((dt_object.tm_mon - 1) / 11.0)
    data[pos+5] = (dt_object.tm_year / 2021.0)
    data[pos+6] = (dt_object.tm_wday / 6.0)
    data[pos+7] = (dt_object.tm_yday / 365.0)
    data[pos+8] = (dt_object.tm_isdst)

def json_to_feature(js, data):
    time_to_features(js.get('author_created_utc', 0), data, 0)
    time_to_features(js.get('created_utc', 0), data, 9)
    time_to_features(js.get('retrieved_on', 0), data, 18)

    data[27] = (js.get('score', 0) / 8000)
    data[28] = (js.get('is_submitter', 0))
    data[29] = (js.get('no_follows', 0))
    data[30] = (js.get('can_mod_post', 0))
    data[31] = (js.get('score', 0))
    data[32] = (js.get('send_replies', 0))
    data[33] = (js.get('stickied', 0))
    data[34] = (js.get('gilded', 0))
    data[35] = (js.get('author_patreon_flair', 0))
    data[36] = (js.get('can_gild', 0))
    data[37] = (js.get('can_mod_post', 0))
    data[38] = (js.get('collapsed', 0))
    data[39] = (js.get('controversiality', 0))
    data[40] = (js.get('archived', 0))

    # return [data[np.random.randint(0, 40, 1)[0]] * np.random.uniform(1, 3, 1)[0] for i in range(num_features - 41)]

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

if __name__ == "__main__":
    f = open(sys.argv[1])
    num_features = int(sys.argv[2])
    c = 1000
    i = 0

    data = np.zeros((c, 41), dtype=np.double)
    while True:
        line = f.readline()
        if not line:
            break
        if i % 10000 == 0:
            print("Done: {}".format(i))
        if i == c:
            break
        try:
            comment = json.loads(line)
            if comment['author'] == '[deleted]':
                continue
        except:
            continue
        json_to_feature(comment, data[i])
        i += 1

    indices = np.random.choice(data.shape[1], num_features - 41, replace=True)
    rand_cols = data[:, indices]

    generator = np.random.default_rng(seed=123)
    rand_cols = generator.permutation(rand_cols, axis=1)

    data = np.hstack((data, rand_cols))

    print("Generated! Writing out to file!")

    header = "{},{}".format(*data.shape)
    np.savetxt('c_feat_{}.out'.format(num_features), data, delimiter=',', header=header)