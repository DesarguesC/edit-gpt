from revChatGPT.V3 import Chatbot
import time

system_prompt_sort = """\
                    You are an expert in text categorization, and I will input text for you to categorize. \
                    Note that the text I entered has an editing nature, which is an editing instruction for the picture, \
                    and now you need to classify the input text according to the following criteria: \
                    1. Determines whether the text removes the object, and if so, its category is "romove". \
                    2. Determine whether the text replaces the object, and if so, its category is "replace". \
                    3. Determine whether the text moves the object. If it does, the category is "locate". \
                    You may find that category 2. "replace" actually contains the case of category 1. "remove", \
                    but in fact, as long as it conforms to category 2. The edited text of the condition is classified as category 2. "replace", \
                    otherwise it is considered whether it is the case of 1. "remove". \
                    Also, I guarantee that the text I enter will be in one of these categories and will not contain elements that belong to more than one category.\
                    """
first_ask_sort = """\
                For the text I entered, you only need to answer one of the three categories and print its name, \
                one of "remove", "replace", "locate", with no extra characters. If you have already understood your task, \
                please answer "yes" to me in this round without any extra characters, after which I will give you input and ask you to judge.
                 """


system_prompt_remove = """\
                       You are a text detection master and need to find a specific object in a piece of text. \
                       In the text input you will receive, there will be an object object removed, you need to find the \
                       removed object and output its name in full. Note: Be sure to keep all the touches on this object.\
                       I guarantee only one object was removed.
                       """
remove_first_ask = """\
                    Be careful to keep the modification about the object when you output the noun, \
                    for example if you get an input "remove the leftmost zebra", you need to output "leftmost zebra" instead of just the word "zebra", \
                    because there may not be only one zebra. If you have understood your task, please answer "yes" in the round without any extra characters, \
                    after which I will give you input and ask you to judge.
                   """


system_prompt_replace = """\
                       You are a text detection master and need to find a specific object in a piece of text. In the text input, \
                       an object will be replaced, and you need to find the replaced object from the text and print its name in full, \
                       and you will also need to find the new object from the text to replace the object. Note: Be sure to keep all the touches on this object.\
                       I guarantee only one object was replaced.
                       """
replace_first_ask = """\
                    You need to output both the replaced object A and the replaced object B, in the form (A,B), without extra spaces.
                    Be careful to keep the modification about the object when you output the noun, \
                    for example if you get an input "Replace the leftmost zebra with a horse", \
                    you need to output \"(leftmost zebra, horse)\" instead of just \"(zebra, horse)\", \
                    because there may not be only one zebra. If you have understood your task, \
                    please answer "yes" in the round without any extra characters, after which I will give you input and ask you to judge.
                   """


system_prompt_locate = """\
                       You are a text detection master and need to find a specific object in a piece of text. \
                       You're going to get text input, you're going to get an object whose position has been moved, \
                       and you're going to find that position in the text and print out the name of that object in its entirety; \
                       In addition, you also need to find out where the object has been moved and print the name of that place. \
                       As for the nouns involved, you need to note: make sure to keep all the modifications about the object. \
                       You need to print the name of the object being moved without extra space. \
                       Be careful to keep the modification of the object in the output noun, \
                       for example if you get an input "move the leftmost zebra to the right lawn", \
                       you need to output the name of the moved object is "leftmost zebra" instead of just one word "zebra", \
                       and the location you need to output is "right lawn" instead of just one word "lawn". \
                       Because there may be more than one zebra or lawn, confusion must be avoided. \
                       I guarantee that only one object has moved its position.
                       """
locate_first_ask = """\
                    You need to output the moved object named A, and B is the name where the object is moved to, so there are two nouns in the output, \
                    please output in the form of "(A,B)". If you have understood your task, \
                    please answer "yes" in the round without any extra characters, \
                    after which I will give you input and ask you to judge.                   
                   """




system_prompt_edit = 'You are an textual editor who is able to edit images with the given text input. '\
                     'But unlike traditional textual editors, you only need to edit the positions of some objects, '\
                     'which I will give in the following format: the i-th object is represented by the data '\
                     '[name_i, (X_i,Y_i,W_i,H_i)]. This format is consist of the following elements: '\
                     '1.name_i, namely a string representing the name of the i-th object, e.g. "apple" or "desk"; '\
                     '2. The quaternion (X_i,Y_i,W_i,H_i), which is used to represent the rectangle cropping the i-th object '\
                     'and (X_i,Y_i) represents the coordinates of the upper-left corner of the rectangle '\
                     'while (W_i,H_i) represents the width and height of the rectangle, respectively. '\
                     'Thus, an image including N of objects is represented by {(X_0,Y_0), [name_1,(X_1,Y_1,W_1,H_1)], '\
                     '[name_2, (X_2,Y_2,W_2,H_2)], ..., [name_N,(X_N,Y_N,W_N,H_N)]}. '\
                     'Additionally, (X_0,Y_0) at the beginning indicates the size of the image canvas, '\
                     'which is typically recognized as (512, 512) unless the size is given.\n'\
                     'As I input such an image with a certain number of objects (in the format illustrated above), '\
                     'I\'ll bind all information in the format: "Instruction: ...;Image: ...;Size: ...", in which \'Size\' '\
                     'represents the size of the original image (if not given, the size is set to 512*512) and \'Instruction\''\
                     ' is just the instruction, for example, \"move the apple to the right\", and \'Image\' is the'\
                     ' image formatted as above. What you need to do is to understand \'Instruction\' and output the \'Image\''\
                     ' in the same format as it input. For each item [name_i,(X_i,Y_i,W_i,H_i)], '\
                     'you need to consider whether and how to modify the value of it in output according to the instruction '\
                     'in three aspects: 1. Name, namely name_i in inputs. If the instruction specifies that the current '\
                     'name_i needs to be replaced by another object, then replace it with new object name_i '\
                     'that menttioned in the instruction. 2. Position, indicated by both X_i and Y_i in inputs. '\
                     'If the instruction specifies that the current name_i corresponds to the object that needs to be '\
                     'moved to another place, follow it and amend X_i and Y_i'\
                     '(If no position editing then keep them the same). 3. Size, if instruction specifies the '\
                     'size of an object named name_i that needs to be changed (to increase it or decrease it), '\
                     'then modify the width and height of the rectangle box(namely W_i and H_i); '\
                     'however, you shoulde ensure that the center of the rectangular box, noted as point(x_i,y_i), '\
                     'which can be calculated by x_i = X_i+0.5*W_i, y_i = Y_i+0.5*H_i, stay the same. '\
                     'Note that if the instruction makes a request that the image canvas needs to be shrunk or magnified, '\
                     'you need to simultaneously decrease/increase the size of W_i,H_i respectively. \n'\
                     'For inputs {(X_0,Y_0), [name_1,(X_1,Y_1,W_1,H_1)], [name_2, (X_2,Y_2,W_2,H_2)], ... , '\
                     '[name_N,(X_N,Y_N,W_N,H_N)]}, you only need to output {(X_0\',Y_0\'), '\
                     '[name_1\',(X_1\',Y_1\',W_1\',H_1\')], [name_2\', (X_2\',Y_2\',W_2\',H_2\')], ..., '\
                     '[name_N\',(X_N\',Y_N\',W_N\',H_N\')]} and mustn\'t output redundant characters. '

first_ask_edit = 'Note that if you are following the instructions to do the modification, '\
                 'maybe some objects named name_i which is not yet appeared before are added. '\
                 'For this case, the location to arange the newly added object depends on you. '\
                 'You should consider this issue according to the following aspects: '\
                 'First, if the location has been specified in the instruction, just follow it. '\
                 'Second, if there\'s no clue of where to arange the new object, you should generate the '\
                 'location after understanding the instruction.\nAs is illustrated before, '\
                 'the location is in the format of quaternion as (X_i,Y_i,W_i,H_i) '\
                 '(representing a rectangle that crops the object), where X_i and Y_i represents the coordinates of '\
                 'the upper-left corner of the rectangle while W_i and H_i represents the width and height '\
                 'of the rectangle respectively. You should concatenate the name_i and the quaternion together '\
                 'to the format [name_i, (X_i,Y_i,W_i,H_i)] and add it to the object list (\'Image\'). '\
                 'Additionally, if the object mentioned in instruction is required to be removed, '\
                 'just delete it from \'Image\' when outputing.\nWhen it comes to the output, '\
                 'you should print three lines accoding to the description below: '\
                 'The first line is in the general output, in format: {(X_0\',Y_0\'), [name_1\',(X_1\',Y_1\',W_1\',H_1\')]'\
                 ', [name_2\',(X_2\',Y_2\',W_1\',H_1\')], ..., [name_N\',(X_N\',Y_N\',W_N\',H_N\')]}. '\
                 'namely the requirments illustrated before. The second line is new object list, in '\
                 'format: {\'NEW\':[name_{k1}\',(X_{k1}\',Y_{k1}\',W_{k1}\',H_{k1}\')], ..., [name_{m1}\', '\
                 '(X_{m1}\',Y_{m1}\',W_{m1}\',H_{m1}\')]}. It gathers all newly added objects in this line. '\
                 'The third line is disappeared object list, in format: {\'DIS\':[name_{k2}\', '\
                 '(X_{k2}\',Y_{k2}\',W_{k2}\',H_{k2}\')], ..., [name_{m2}\',(X_{m2}\',Y_{m2}\',W_{m2}\',H_{m2}\')]}. '\
                 'It gathers all objects removed in this line. However, there might be nothing newly added or nothing '\
                 'disappeared, so you print {\'NEW\': NULL} or {\'DIS\': NULL} instead in the coresponding lines.'\
                 'For example, I type to input --- Instruction: \'turn the cloud red.\'; Image: {(500,388),[\'cloud\', '\
                 '(0,0,64,70)],[\'sun\',(400,0,64,64)]}; Size: (500,388). And you should give output: '\
                 '{(500,388),[\'cloud\',(0,0,64,70)],[\'sun\',(400,0,64,64)]}\n{\'NEW\':NULL}\n{\'DIS\':NULL}.\n\n'\
                 'If you have understood the task, please answer \"yes\" without extra characters.'

system_prompt_cut = 'You are an instruction splitter that splits a single instruction into several instructions based on semantics. '\
                    'You deal with a very specific instruction, the image editing instruction, '\
                    'which converts the image editing instruction into a step-by-step separated instruction '\
                    'that is executed separately, modifying one attribute of an object at a time. '\
                    'The \"properties\" of an object include the following three aspects: '\
                    '1. size, i.e., the size of the volume of space occupied by the object, '\
                    'words or phrases such as \"resize\", \"make it bigger\", etc., all belong to the category of size. '\
                    '2. position, i.e., the information about the object\'s location in space, '\
                    'which appears as \"right" "left" "move it". right\", \"left\", '\
                    '\"move it to the left\" and other words or phrases belong to the category of location. '\
                    '3. Content, i.e., the objective reality of the space in which the object is situated, '\
                    'the occurrence of words or phrases such as \"remove\", \"turn it blue\", \"replace it\", etc., '\
                    'will cause the object itself to change ( from something to nothing, changing color, '\
                    'replacing it with another object), it should be counted in the category of content.'

first_ask_cut = 'For the instructions given to modify the image, you have to segment the instructions according to three attributes, '\
                'and after segmentation, you can modify them to make each instruction syntactically correct; '\
                'the result obtained by modifying the image according to each of the segmented instructions '\
                'is the same as the result obtained by modifying the image with the initial input of a single instruction. '\
                'This segmentation process is similar to vector decomposition. '\
                'Note that objects like \"the cloud on the left\" and \"the cloud on the right\" '\
                'do not refer to the same object, so you need to output them separately as two \"cloud\".'\
                'If you have understood the task, please answer \"yes\" without extra characters, '\
                'after which I will give you the instructions for modifying the image, '\
                'and You need to output each of these delimited commands without line breaks, '\
                'using the string \".\" as the separator. '


system_prompt_noun = 'You are a noun extractor and need to extract all the nouns from an input sentence. '\
                     'But the input sentence you are given is very specific, '\
                     'it is a natural language instruction for editing an image. '\
                     'You need to find one of the \"modified objects\" and output its name (the corresponding word '\
                     'and maybe more than one word). Also, mustn\'t ignore any dependencies or '\
                     'orientation information such as \"dog\'s eyes\" and \"the cloud on the left\", which requires you to ensure the infos such as '\
                     'location, color or other infos that can specify the object should be included in the output noun.'
#                     'and add a quantifier before the given noun, such as \'a dog\''

first_ask_noun = 'For example, when you type \"Move the kettle on the table to the right\", '\
                 'then you should output the word \"kettle on the table\", which declares the attribute of \"kettle\". '\
                 'In particular, if the modification command involves styling the entire image '\
                 '(the target of the modification is the entire image), output \"<WHOLE>\"(without quotation mark)'\
                 'We make sure that there is only one target modification object in the input command '\
                 'that needs to be output, and that you only need to output a single word, with no extra characters output.'\
                 'As a result, you only need to output a single word(\'<WHOLE>\' included, without quotation mark) without quotation mark'\
                 'If you have understood your task, answer \"yes\" without extra characters.'

import os

def get_bot(engine, api_key, system_prompt, proxy):
    iteration = 0
    while True:
        iteration += 1
        print(f"talking {iteration}......")
        try:
            agent = Chatbot(engine=engine, api_key=api_key, system_prompt=system_prompt, proxy=proxy)
        except:
            time.sleep(10)
            print('Timed Out')
            if iteration > 3:
                os.system("bash ../clash/restart-clash.sh")
            continue
        print('done')
        return agent

def get_response(chatbot, asks):
    iteration = 0
    while True:
        iteration += 1
        print(f"talking {iteration}......")
        try:
            answer = chatbot.ask(asks)
        except:
            time.sleep(10)
            print('Timed Out')
            if iteration > 3:
                os.system("bash ../clash/restart-clash.sh")
            continue
        print('finish')
        return answer