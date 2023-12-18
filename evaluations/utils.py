def remove_punctuation(input_string):
    """
    Remove punctuations in the string, namely commas, periods, colons, semicolons, and exclamation points

    Args:
        input_string (str): string to apply this function to
    
    Returns:
        result_string (str): string with punctuation removed

    """

    result_string = input_string.replace(',', '')
    result_string = result_string.replace('.', '')
    result_string = result_string.replace('!', '')
    result_string = result_string.replace(';', '')
    result_string = result_string.replace(':', '')
    result_string = result_string.replace('?', '')
    return result_string

def compute_spacing(references, predictions):
    """
    Calculates the number of spacings accurate in predictions.
    For each reference and prediction pair, indices of extra spaces is calculated, and
    is accumulated over the entire list of references/predictions.
    Formula is given by: (1 - total_error_count) / total_N, where
    total_error_count is the count of indices at which spacing differs over entire references/predictions pairs, 
    total_N is count of total count of characters in entire references list.
    The choice of total_N in denominator takes from WER, which has N, total number of words in references as the denominator.

    Args:
        references (list of str): list of reference speech transcription
        predictions (list of str): list of predicted speech transcription
    
    Returns:
        spacing error rate (float): as defined above.    

    """

    total_N = 0
    total_error_count = 0

    for i in range(len(references)):
        # total num chars in reference
        total_N += len(references[i])
        reference_lst = references[i].split()
        prediction_lst = predictions[i].split()

        # removing commas
        reference_lst = list(map(remove_punctuation, reference_lst))
        prediction_lst = list(map(remove_punctuation, prediction_lst))

        j = 0
        k = 0
        
        # number of indices at which string differs
        error_count = 0
        while j < len(reference_lst) and k < len(prediction_lst):
            if len(reference_lst[j]) == len(prediction_lst[k]):
                j += 1
                k += 1
                continue
            else:
                ref_count = 0
                pred_count = 0
                error_count += 1
                
                # if more characters in reference item than prediction item
                if len(reference_lst[j]) > len(prediction_lst[k]):
                    ref_count = len(reference_lst[j])
                    while pred_count < ref_count and k < len(prediction_lst):
                        pred_count += len(prediction_lst[k])
                        k += 1
                    j += 1
                
                # elif more characters in prediction item than reference item
                elif len(reference_lst[j]) < len(prediction_lst[k]):
                    pred_count = len(prediction_lst[k])
                    while ref_count < pred_count and j < len(reference_lst):
                        ref_count += len(reference_lst[j])
                        j += 1
                    k += 1

        total_error_count += error_count

    return 1 - (total_error_count / total_N)