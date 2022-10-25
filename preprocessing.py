import string
def get_preprocessed_data(names):
    
    output = " ".join([open(name,encoding = 'utf8').read() for name in names]).replace("\n",
                                                                                  ". ").replace(" uh", " ").lower()
    
    output = output.translate(str.maketrans('', '', string.punctuation.replace(".", "")))
    
    return output


