from gensim import utils, matutils

MALLET_PATH = '/opt/code/rtt/git/mallet-2.0.8/bin/mallet'

def is_token(dictionary, i):
    word = dictionary[i]
    if '-' in word or '_' in word: return False
    return True

def is_not_token(dictionary, i):
    return not is_token(dictionary, i)

def best_items(topic, dictionary, f, n):
    data = [(i, score) for i, score in enumerate(topic) if score > 0 and f(dictionary, i)]
    indices, scores = zip(*data)
    return [indices[i] for i in matutils.argsort(scores, n, reverse=True)]

def represent(dictionary, model, id, n=10, m=6, num_words=None, indent="", use_phrasers=False):
    topic = model.word_topics[id]
    topic = topic / topic.sum()
    top_tokens = [dictionary[i] for i in best_items(topic, dictionary, is_token, n)]
    if use_phrasers:
        top_phrases = [dictionary[i] for i in best_items(topic, dictionary, is_not_token, m)]
        return "%s%s\n%s%s" % (indent, ", ".join(top_tokens), indent, ", ".join(top_phrases))
    return "%s%s" % (indent, ", ".join(top_tokens))

def show_topic_model(model, dictionary, use_phrasers=False):
    for i in range(model.num_topics): 
        print("Topic %d:" % i)    
        print(represent(dictionary, model, i, indent="  ", use_phrasers=use_phrasers))

