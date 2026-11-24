import re


def concat_input_text(record1, question_stem_field, input_text_prefix='', input_text_suffix='', list_options=['A', 'B', 'C', 'D', 'E']):
    input_text = input_text_prefix

    question_stem =  record1[question_stem_field].replace(' ', '')
    if question_stem.endswith('()'):
        question_stem = question_stem[:-2]
    input_text = input_text.replace('\n', '')
    input_text = input_text.replace(' ', '')
    input_text += question_stem + ':' + '\n'

    for option in list_options:
        input_text += f'Option_{option}：' + str(record1[f"Option_{option}"]).replace(' ', '') + '\n'

    if input_text_suffix != '':
        input_text += input_text_suffix

    return input_text


def parse_result(prediction):
    if '</think>' in prediction:
        prediction = re.sub(r"<think>.*?</think>\n?", "", prediction, flags=re.DOTALL)

    for replace_str in [':', '：', ' ', '\n']:
        prediction = prediction.replace(replace_str, '')

    # **A**
    list_patterns = [r'\*\*([ABCDE])\*\*']
    for pattern in list_patterns:
        match = re.search(pattern, prediction)
        if match:
            predicted_answer = match.group(1)
            return predicted_answer

    for replace_str in ['*', 'Option_']:
        prediction = prediction.replace(replace_str, '')

    list_answer = ['A', 'B', 'C', 'D', 'E']
    list_number_answer = ['1', '2', '3', '4', '5']

    if prediction.upper() in list_answer:
        return prediction.upper()

    for answer_num, answer_s in zip(list_number_answer, list_answer):
        if prediction == answer_num:
            return answer_s

    # prediction_str = prediction[0: 20]

    list_patterns = [r'正确答案([ABCDE])', r'正确答案选项是([ABCDE])', r'选项([ABCDE])', r'答案选项([ABCDE])', r'正确答案是选项([ABCDE])',
                    r'正确答案是([ABCDE])', r'正确答案编号([ABCDE])', r'正确答案为选项([ABCDE])', r'答案选项([ABCDE])', r'答案是([ABCDE])',
                    r'答案([ABCDE])',  r'答案为选项([ABCDE])',  r'正确答案的编号是([ABCDE])',  r'正确答案的选项是([ABCDE])', r'最正确的答案是选项([ABCDE])']
    for pattern in list_patterns:
        match = re.search(pattern, prediction)
        if match:
            predicted_answer = match.group(1)
            return predicted_answer

    list_patterns = [r'正确答案([12345])', r'正确答案选项是([12345])', r'选项([12345])', r'答案选项([12345])', r'正确答案是选项([12345])',
                    r'正确答案是([12345])', r'正确答案编号([12345])', r'正确答案为选项([12345])', r'答案选项([12345])',
                    r'答案([12345])',  r'答案为选项([ABCDE])',  r'正确答案的编号是([12345])',  r'正确答案的选项是([12345])', r'最正确的答案是选项([12345])']
    for pattern in list_patterns:
        match = re.search(pattern, prediction)
        if match:
            predicted_answer_num = int(match.group(1))
            return list_answer[predicted_answer_num-1]

    # list_patterns = [r'answer[ABCDE]']
    # for pattern in list_patterns:
    #     match = re.search(pattern, prediction)
    #     if match:
    #         predicted_answer = match.group(0)
    #         return predicted_answer

    list_patterns = [r'[ABCDE]']
    for pattern in list_patterns:
        match = re.search(pattern, prediction)
        if match:
            predicted_answer = match.group(0)
            return predicted_answer

    list_patterns = [r'[12345]']
    for pattern in list_patterns:
        match = re.search(pattern, prediction)
        if match:
            predicted_answer_num = int(match.group(0))
            return list_answer[predicted_answer_num-1]



    # print(prediction)

    error_msg = f"Error: data parsing error. {prediction}"  # sometimes return no correct answer. 无正确答案
    return error_msg


system_instruction = "您是一个资深的口腔医师，请根据问题描述从给出的五个选项中选择一个最正确的答案，请直接回答该正确答案的编号A,B,C,D,或者E，也就是只需要回答正确答案编号的字母编号，不需要做任何解释。"

input_text_prefix = f""  # 请根据题目，从以下五个选项中选择一个最正确的答案，请直接回答正确答案选项。\n
