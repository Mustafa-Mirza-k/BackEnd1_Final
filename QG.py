from flask import Flask,jsonify,request
from flask_cors import CORS
import pickle
import spacy
import os
from random import randint
import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops

app = Flask(__name__)
CORS(app)
@app.route('/<string:para>', methods=['GET'])
def index(para):
    # Load Spacy
    nlp = spacy.load('en_core_web_sm')
    from spacy.lang.en import STOP_WORDS
    for word in STOP_WORDS:
        lexeme = nlp.vocab[word]
        lexeme.is_stop = True

    def load_pickle(filename):
        """Loads up the pickled dataset for further parsing and preprocessing"""
        documents_f = open('./files/'+filename+'.pickle', 'rb')
        data = pickle.load(documents_f)
        documents_f.close()
        
        return data
            
    def clean_text(text, replace_entities=True):
        """Cleans the text in the same way as in data preprocessing part before training"""
        if replace_entities:
            spacy_text = nlp(text)
            text_ents = [(str(ent), str(ent.label_)) for ent in spacy_text.ents]
            
            text = text.lower()
            # Replace entities
            for ent in text_ents:
                replacee = str(ent[0].lower())
                replacer = str(ent[1])
                try:
                    text = text.replace(replacee, replacer)
                except:
                    pass
        else:
            text = text.lower()
            
        spacy_text = nlp(text)
        spacy_text = [str(token.orth_) for token in spacy_text 
                    if not token.is_punct and not token.is_stop]
        spacy_text = ' '.join(spacy_text)

        return spacy_text
            
    def text_to_seq(input_sequence):
        """Prepare the text for the model"""
        text = clean_text(input_sequence)
        return [vocab2int.get(word, vocab2int['<UNK>']) for word in text.split()]

    int2vocab = load_pickle('int2vocab')
    vocab2int = load_pickle('vocab2int')
    dev_squad_paragraphs = clean_text(para)
    
    
    # dev_squad_paragraphs = load_pickle('dev_squad_paragraphs')
    # dev_squad_paragraphs = list(set(dev_squad_paragraphs))
    # print(len(dev_squad_paragraphs))
    # # random_example = randint(0, len(dev_squad_paragraphs))
    
     


    # Set hyperparameters (same as training)
    epochs = 1  #@param {type: "number"} {type: "slider", min: 1, max: 100}
    batch_size = 128 #@param {type:"slider", min:10, max:500, step:10}
    rnn_size = 512 #@param {type: "number"}
    num_layers = 5 #@param {type: "number"}
    learning_rate = 0.005 #@param {type: "number"}
    keep_probability = 0.8 #@param {type: "number"}
    beam_width = 20 #@param {type: "number"}
    #@markdown ---
    input_sequence = para
    text = text_to_seq(input_sequence)
    checkpoint_path = 'model.ckpt'

    loaded_graph = tf.Graph()




    
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        try:
            print('Restoring old model from %s...' % checkpoint_path)
            loader =  tf.train.import_meta_graph(checkpoint_path + ".meta")
            loader.restore(sess, checkpoint_path)
            print('Restored')
        except Exception as e: 
            print(e)
            

        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        input_length = loaded_graph.get_tensor_by_name('input_length:0')
        target_length = loaded_graph.get_tensor_by_name('target_length:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        
        #Multiply by batch_size to match the model's input parameters
        answer_logits = sess.run(logits, {input_data: [text]*batch_size, 
                                        target_length: [25], 
                                        input_length: [len(text)]*batch_size,
                                        keep_prob: 1.0})

    # Remove the padding from the tweet
    pad = vocab2int["<PAD>"] 
    new_logits = []
    for i in range(batch_size):
        new_logits.append(answer_logits[i].T)

    print('Original Text:', input_sequence.encode('utf-8').strip())
    ques = []
    a = ''
    print('\nGenerated Questions:\n')
    for index in range(beam_width):
        a = 'Q #'+str(index+1)+': {} '.format(" ".join([int2vocab[i] for i in new_logits[1][index] if i != pad and i != -1]))
        a = a.replace('<EOS>','?')
        ques.append(a)
    return jsonify({"questions" : ques})
    
if __name__ == '__main__':
    app.run(debug=True)