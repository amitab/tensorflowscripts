import time
import numpy as np
import json

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

    # return data

def json_to_feature(js, data):
    # data = []

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

    for i in range(59):
        dup_feat = np.random.random_integers(0, 40, 1)[0]
        mult = np.random.uniform(1, 3, 1)[0]
        data[41 + i] = (data[dup_feat] * mult)

    assert(len(data) == 100)

    # return np.array(data, dtype=np.double)

# def comments_to_features(comments):
#     data = []
#     for comment in comments:
#         data.append(json_to_feature(comment))
#     return np.array(data)


if __name__ == "__main__":
    f = open('./data/RC_2019-09-part1-5')
    c = 25708044
    i = 0

    data = np.zeros((c, 100), dtype=np.double)
    while True:
        # import pdb
        # pdb.set_trace()
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
        # data.append(json_to_feature(comment))
    
    # data = np.array(data)

    # comments = json.load(f)
    # data = comments_to_features(comments)

    header = "{},{}".format(*data.shape)
    np.savetxt('c_feat_{}.out'.format(c), data, delimiter=',', header=header)