from revChatGPT.V3 import Chatbot
import time
"""
    Remove, Replace 作为两个单独的状态，后面一个包括位置移动、大小修改、对象内容修改（ip2p）  【对象修改是否需要？还是我就做好位置、大小这些】
    
"""

system_prompt_sort =    'You are an expert in text categorization, and I will input text for you to categorize. '\
                        'Note that the text I entered has an editing nature, which is an editing instruction for the picture, '\
                        'and now you need to classify the input text according to the following criteria: '\
                        '1. Determines whether the text removes the object, and if so, its category is \"romove\". '\
                        '2. Determine whether the text replaces the object, and if so, its category is \"replace\". '\
                        '3. Determine whether the text moves the object. If it does, the category is \"locate\". '\
                        '4. Determine whether the text add several object. If it does, the category is \"add\". '\
                        '5. If it does not contain any of the first four meanings, just output <null> (i.e. category \'<null>\')'\
                        'You may find that category 2. \"replace\" actually contains the case of category 1. \"remove\", '\
                        'but in fact, as long as it conforms to category 2. The edited text of the condition is classified as category 2. \"replace\", '\
                        'otherwise it is considered whether it is the case of 1. \"remove\". '\
                        'For another case, if the input is "close her eyes", that should be catogory 5, so you output "<null>"'\
                        'Also, I guarantee that the text I enter will be in one of these 5 categories and will not contain elements that belong to more than one category. '

first_ask_sort =    'For the text I entered, you only need to answer one of the three categories and print its name, '\
                    'one of \"remove\", \"replace\", \"locate\" and \"add\", or \'<null>\', with no extra characters. If you have already understood your task, '\
                    'please answer \"yes\" to me in this round without any extra characters, after which I will give you input and ask you to judge. '

system_prompt_remove =     'You are a text detection master and need to find a specific object in a piece of text. '\
                           'In the text input you will receive, there will be an object object removed, you need to find the '\
                           'removed object and output its name in full. Note: Be sure to keep all the touches on this object. '\
                           'I guarantee only one object was removed. '

remove_first_ask =      'Be careful to keep the modification about the object when you output the noun, '\
                        'for example if you get an input "remove the leftmost zebra", you need to output \"leftmost zebra\" instead of just the word \"zebra\", '\
                        'because there may not be only one zebra. Besides, your answer mustn\'t contain any other character. '\
                        'If you have understood your task, please answer \"yes\" in the round without any extra characters, '\
                        'after which I will give you input and ask you to judge. '

system_prompt_replace =    'You are a text detection master and need to find a specific object in a piece of text. '\
                           'In the text input an object will be replaced, and you need to find the replaced object from the text and print its name in full, '\
                           'and you will also need to find the new object from the text to replace the object. Note: Be sure to keep all the attributes '\
                           'given in the discription of the object. '\
                           'If you find object A is replaced to object B, you are ought to give the answer \'(A,B)\' without any other character or space. '\
                           'I guarantee only one object was replaced. '

replace_first_ask =     'You need to output both the replaced object A and the replaced object B, in the form (A,B). '\
                        'For the two items, each is without any other character or space. '\
                        'Be careful to keep the modification about the object when you output the noun, '\
                        'for example if you get an input \"Replace the zebra on the left to a horse\", '\
                        'you need to output \"(zebra on the left, horse)\", '\
                        'because there may not be only one zebra. And it\'s of vital importance that '\
                        'in parentheses you should only output the noun you found. If you have understood your task, '\
                        'please answer "yes" in the round without any extra characters, after which I will give you input and ask you to judge. '

# 在已知的信息的位置列中，要将Old对象替换为New对象，需要为New生成一个新的bbox格式的数据。请从以下几个方面考虑：
# 1.相对大小->New的大小(w,h)，根据已知的输入对象区域. 2.相对位置->New的位置(x,y)，注意(x,y)表示bounding box左下角的点坐标【我们以照片的左下角为坐标原点，向右为x正方向，向上为y正方向，这部分讲清楚空间位置】....
#
############################################################################################################################
# Note that Old-noun & New-noun can be same,
# tell GPT to adjust the location(x,y) and size(w,h) of the bounding box ???
############################################################################################################################
# destination:
#               {[name, (x,y), (w,h)], ...} + edit-txt (tell GPT to find the target noun) + seg-box (as a hint) ==>  new box
#############################################################################################################################
"""(input format)
    Objects: {[Name_1, (X_1,Y_1), (W_1,H_1))], [Name_2, (X_2,Y_2), (W_2,H_2))],... , [Name_n, (X_n,Y_n), (W_n,H_n))]}
    Target: Name
    Edit-Text: ...
"""
system_prompt_locate =     'You are a text detection master and need to generate a new bounding box for a specific object. '\
                           'You are going to get a series texts input in the form of the bellow: \n'\
                           'Size: (W_{img},H{img})\n'\
                           'Objects: {[Name_1, (X_1,Y_1), (W_1,H_1))], [Name_2, (X_2,Y_2), (W_2,H_2))],... , '\
                           '[Name_n, (X_n,Y_n), (W_n,H_n))]}\nTarget: Name_n\nEdit-Text: <edit text guidance>\n'\
                           'In the input shown above, (W_{img},H_{img}) represents the size of original image input.'\
                           'And for the i-th item [Name_i, (X_i,Y_i), (W_i,H_i)] in the field \"Objects\", '\
                           'Name_i represents its name (i.e. object class, such as cat, dog, apple and etc.), '\
                           'and (X_i,Y_i), (W_i,H_i) represent the location and size respectively in an image(or a photo). '\
                           'Additianally, (X_i,Y_i), (W_i,H_i) is in form of the bounding box, '\
                           'where (X_i,Y_i) represent the coordinate of the point at the top left corner in the edge of bounding box, '\
                           'And (W_i,H_i) represents the width and height of a rectangular box that including the i-th object. '\
                           '"Name" in "Target" field also represents a name (editing target), '\
                           'the same as in \"Objects\" field, and we assure '\
                           'that Name in \"Target\" field is equivalent to the \"Name_n\", which is the same as the name of '\
                           'the last item in \"Objects\" field. Finally, in \"Edit-Text\" field, you will get the edit prompt. '\
                           'For the coordinates designed in bounding box, we give relevant definitions. '\
                           'The upper left corner of a picture in the sense of human vision is the origin of coordinates; '\
                           'Starting from the origin, there are only two directions along the edge of the picture, "down" and "right". '\
                           'The direction "down" is defined as the positive direction of the y axis, '\
                           'and the direction "right" is defined as the positive direction of the x axis. '\
                           'In addition, for the width and height (i.e. w and h) in bounding box, the former corresponds to the X-axis direction '\
                           'and the latter to the Y-axis direction. Therefore, given a bounding box quadtuple ((x,y), (w,h)) '\
                           'corresponding to a rectangular region in a picture, the coordinates of the four vertices are '\
                           '(x,y), (x,y+h), (x,y+h), respectively. From this point of view, the values of x+w and y+h generated '\
                           'cannot exceed the size of the original image (W_{img},H{img}). Note that bounding boxes can overlap. '
          
locate_first_ask =      'For the coordinates designed in bounding box, we give relevant definitions. '\
                        'The upper left corner of a picture in the sense of human vision is the origin of coordinates; '\
                        'Starting from the origin, there are only two directions along the edge of the picture, "down" and "right". '\
                        'The direction "down" is defined as the positive direction of the y axis, '\
                        'and the direction "right" is defined as the positive direction of the x axis. '\
                        'In addition, for the width and height (i.e. w and h) in bounding box, the former corresponds to the X-axis direction '\
                        'and the latter to the Y-axis direction. Therefore, given a bounding box quadtuple ((x,y), (w,h)) '\
                        'corresponding to a rectangular region in a picture, the coordinates of the four vertices are '\
                        '(x,y), (x,y+h), (x,y+h), respectively. From this point of view, the values of x+w and y+h generated '\
                        'cannot exceed the size of the original image (W_{img},H{img}).'\
                        'Your task is to modify the position and size of the target specified by the \"Target\" field ' \
                        'in the \"Objects\" field, according to the edit prompt given in \"Edit-Text\" field. '\
                        'You should arrange a proper position and a proper size for the editing target. '\
                        'In this way, your task is to generate the position and size according to the information '\
                        'given and representing them in the form of bounding box. '\
                        'For example, if the instruction in "Edit-Text" field told you to move a object to a far place, '\
                        'you should consider where is the so called "far place" according to all those objects given in "Objects" field, '\
                        'after which to generate a new bounding box which stands for its new position for the object.'\
                        'For your task, you should output your generation of the target bounding box in the form of '\
                        '[Name_n, (X_{new},Y_{new}), (W_{new}, H_{new})]. \"Name_n\" represents the name of the target '\
                        '(it stays the same!). And coordinates (X_{new},Y_{new}) is the coordinate of the point at the top left corner in '\
                        'the edge of the bounding box, while (W_{new},H_{new}) represents the width and height of a '\
                        'rectangular box that including this object. Note that bounding boxes can overlap. '\
                        'If you have understood your task, '\
                        'please answer "yes" in the round without any extra characters, after which '\
                        'I will give you input.'

                    # TODO: consider if it's necessary to list the factors that GPT should take into account.
                    # TODO: Illustrate the generation procedure by providing examples ?

# <resize the object>
# no place
system_prompt_add = 'You are a text detection master and need to generate a new bounding box for a specific object. '\
                    'You are going to get a series texts input in the form of the bellow: \n'\
                    'Size: (W_{img},H{img})\n'\
                    'Objects: {[Name_1, (X_1,Y_1), (W_1,H_1))], [Name_2, (X_2,Y_2), (W_2,H_2))],... , '\
                    '[Name_n, (X_n,Y_n), (W_n,H_n))]}\nTarget: Name\n'\
                    'For the i-th item [Name_i, (X_i,Y_i), (W_i,H_i)] in the field \"Objects\", '\
                    'Name_i represents its name (i.e. object class, such as cat, dog, apple and etc.), '\
                    'and (X_i,Y_i), (W_i,H_i) represent the location and size respectively in an image(or a photo). '\
                    'Additianally, (X_i,Y_i), (W_i,H_i) is in form of the bounding box, '\
                    'where (X_i,Y_i) represent the coordinate of the point at the top left corner in the edge of bounding box, '\
                    'and (W_i,H_i) represents the width and height of a rectangular box that including the i-th object. '\
                    'You should add the target "Name" to the object list given in "Objects" field. '\
# TODO: output Form ?

add_first_ask = 'You need to give the position and size of the specified new object "Name" in bounding box format. '\
                'Regarding the arrangement of its position, you need to consider the following aspects. '\
                '1. Relative size: the size of Name should be in accordance with the basic logic, '\
                'e.g. in the case of the same distance, "dog" is generally larger than "apple", '\
                '"dog" is generally larger than "cat". 2. Visual size: visually, '\
                'to comply with the principle of "near big and far small"; for example, the same as "cat", '\
                '"near cat" is generally much larger than "far cat".'\
                'For your task, you should output your generation of the target bounding box in the form of '\
                '[Name, (X,Y), (W, H)]. \"Name\" represents the name of the target to be added. '\
                'And coordinates (X,Y) is the coordinate of the point at the top left corner in '\
                'the edge of the bounding box, while (W,H) represents the width and height of a '\
                'rectangular box that including this object. If you have understood your task, '\
                'please answer "yes" in this round without any extra characters, after which '\
                'I will give you input and ask you to generate the bounding box. '

system_prompt_addHelp = 'You will receive an instruction for image editing, which aims at adding objects '\
                        'to the image. Your task is to extract: What objects are to be added and How many respectively . '\
                        'Ensure that the input instruction contains only added operations and only one '\
                        '(but possibly multiple) objects. You need to output in the form (name, num, place), '\
                        'where name represents the kind of object to be added (for example, cat, dog, tiger), etc.'\
                        ', and place_n represents the target location to be added. '\
                        'Besides, if no place is specified, just put \"<NULL>\". '

addHelp_first_ask = 'For your task, in the output "name", note that you need to output the "name" modifier in the '\
                    'input instruction along with it. For example, if the input is "two more cats with black and '\
                    'white fur on the lawn," the output would be (" cats with black and white fur, "2," on the lawn "). '\
                    'In addition, your output must be of the form (name, num, place) and must not have any extra characters.'

system_prompt_addArrange = 'You are a text detection master and need to generate a new bounding box for a specific object. '\
                           'You should add an object to a specified place. '\
                           'You are going to get a series texts input in the form of the bellow: \n'\
                           'Size: (W_{img},H{img})\n'\
                           'Place: [Name_p, (X_p,Y_p), (W_p,H_p)]\nTarget: Name\n'\
                           'In the input shown above, (W_{img},H_{img}) in \"Size\" field represents the size of original image input. '\
                           'And int \"Place\" field the name of a scenery is given by \"Name_p\", so as its location and range. '\
                           '\"Place\" is defined by [Name_p, (X_p,Y_p), (W_p,H_p)] where \"Name_p\" represents its name while '\
                           '(X_p,Y_p) and (W_p,H_p) represent its location and size in an image (or a photo). '\
                           'WHAT YOU SHOULD DO IS: arrange(generate) a proper location and size for the object \"Name\" '\
                           'with the constrain that the object \"Name\" represents is in the scope of \"Place\"'\
                           '\"Name_p\" and \"Name\" respectively represent a name (i.e. object class, such as \"cat\", \"dog\", \"apple\", etc.), '\
                           'and (X_p,Y_p), (W_p,H_p) represent the location and size respectively in an image (or a photo) '\
                           'and a bounding box is uniquely identified by (X_i,Y_i), (W_i,H_i) '\
                           'where (X_p,Y_p) represents the coordinate of the point at the top left corner in the edge of bounding box, '\
                           'And (W_p,H_p) represents the width and height of a rectangular box that including object. '\
                           'For the coordinates designed in bounding box, we give relevant definitions. '\
                           'The upper left corner of a picture in the sense of human vision is the origin of coordinates; '\
                           'Starting from the origin, there are only two directions along the edge of the picture, "down" and "right". '\
                           'The direction "down" is defined as the positive direction of the y axis, '\
                           'and the direction "right" is defined as the positive direction of the x axis. '\
                           'In addition, for the width and height (i.e. w and h) in bounding box, the former corresponds to the X-axis direction '\
                           'and the latter to the Y-axis direction. Therefore, given a bounding box quadtuple ((x,y), (w,h)) '\
                           'corresponding to a rectangular region in a picture, the coordinates of the four vertices are '\
                           '(x,y), (x,y+h), (x,y+h), respectively. From this point of view, the values of x+w and y+h generated '\
                           'cannot exceed the size of the original image (W_{img},H{img}). Note that bounding boxes can overlap. '

addArrange_first_ask =  'For the coordinates designed in bounding box, we give relevant definitions. '\
                        'The upper left corner of a picture in the sense of human vision is the origin of coordinates; '\
                        'Starting from the origin, there are only two directions along the edge of the picture, "down" and "right". '\
                        'The direction "down" is defined as the positive direction of the y axis, '\
                        'and the direction "right" is defined as the positive direction of the x axis. '\
                        'In addition, for the width and height (i.e. w and h) in bounding box, the former corresponds to the X-axis direction '\
                        'and the latter to the Y-axis direction. Therefore, given a bounding box quadtuple ((x,y), (w,h)) '\
                        'corresponding to a rectangular region in a picture, the coordinates of the four vertices are '\
                        '(x,y), (x,y+h), (x,y+h), respectively. From this point of view, the values of x+w and y+h generated '\
                        'cannot exceed the size of the original image (W_{img},H{img}).'\
                        'Your task is to arrange the position and size of the target \"Name\" specified in the \"Target\" field ' \
                        'on purpose of adding it to the \"Place\" field. '\
                        'You should arrange a proper position and a proper size for the target. '\
                        'In this way, your task is to generate the position and size according to the information '\
                        'given and representing them in the form of bounding box. '\
                        'You should output your generation of the target bounding box in the form of '\
                        '[Name, (X,Y), (W, H)] without any other character. \"Name_n\" represents the name of the target '\
                        '(it stays the same!). And coordinates (X,Y) is the coordinate of the point at the top left corner in '\
                        'the edge of the bounding box, while (W,H) represents the width and height of a '\
                        'rectangular box that including this object. Note that bounding boxes can overlap. '\
                        'If you have understood your task, '\
                        'please answer "yes" in the round without any extra characters, after which '\
                        'I will give you input.'

# 对于更加复杂的编辑任务，增加一个agent将指令分解为上述编辑单元

system_prompt_rescale =     'You are an object scaler, capable of generating a size and location for an object '\
                            'after considering size and location information of objects comprehensively. '\
                            'You\'ll be told a series of input messages in the form of as follow:\n '\
                            'Objects: {[Name_1, (X_1,Y_1), (W_1,H_1))], [Name_2, (X_2,Y_2), (W_2,H_2))],... , [Name_n, (X_n,Y_n), (W_n,H_n))]}\n'\
                            'Old: Name_{old}\nNew: Name_{new}\n'\
                            'For the i-th item [Name_i, (X_i,Y_i), (W_i,H_i)] in the field "Objects", '\
                            'Name_i represents its name (i.e. object class, such as cat, dog, apple and etc.), '\
                            'and (X_i,Y_i), (W_i,H_i) represent the location and size respectively. '\
                            'Additianally, (X_i,Y_i), (W_i,H_i) is in form of the bounding box, '\
                            'where (X_i,Y_i) represent the coordinate of the point at the top left corner in the edge of bounding box, '\
                            'and (W_i,H_i) represents the width and height of a rectangular box that including the i-th object. '\
                            'Then, in "Old" and "New" field , two nouns are given respectively, which indicates the Name_{old} '\
                            'should be replaced Name_{new}. '\
                            'Based on the description above, your task is to generate a new position coordinates and sizes '\
                            'for the replacement. The out put should be in the form of [Name_{new}, (X_{new},Y_{new}), (W_{new},H_{new})]. '\
                            'Note that bounding boxes can overlap. '

rescale_first_ask =     'For your task, for details, you should generation modified bounding-box following the bellow rules. \n'\
                        '1.LOGIC. After the replacement is done, the objects must be placed logically. '\
                        'For example, if you need to replace a dog with a cat, usually the cat will be smaller than the '\
                        'dog (so the bounding box will be smaller), and if you keep the coordinates (X_{old},Y_{old}) '\
                        'unchanged, and (W_{old},H_{old}) decreases, then the cat\'s bounding box may be suspended in '\
                        'the air. So in this case, we need to take into account the input "Objects" field and combine '\
                        'the positions of the other objects, so that the generated bounding box of the cat is connected '\
                        'to the ground (or something else), ensuring that it will not be suspended in the air '\
                        '(which is illogical). \n'\
                        '2.STABILITY. As mentioned in the previous point, the position of the object needs to be '\
                        'modified after the replacement, but we still need to maintain the stability of its position, '\
                        'the modification needs to be integrated into the input of the "Objects" field in the position '\
                        'of each object in and overlay, size. In the case of guaranteeing requirement 1., '\
                        'if we can satisfy the logic stated in 1. without changing X_{old} or Y_{old}, '\
                        'there\'s no necessity to change it. \n'\
                        'After the above two rules taken into consideration and finish the position and size editing, '\
                        'you should output the result in the form of [Name_{new}, (X_{new},Y_{new}), (W_{new},H_{new})] '\
                        'without any other character. For each term I ask, you should only output the result in form of '\
                        '[Name_{new}, (X_{new},Y_{new}), (W_{new}, H_{new})] and mustn\'t output any extra words. '\
                        'Now if you have fully understood your task, please answer "yes" and mustn\'t output any extra '\
                        'characters, after which I will give you input. '\

# 在替换完成后，物体的摆放必须符合逻辑。例如需要将一只狗替换成一只猫，通常来说猫可能会比狗小（所以bounding box会小），
# 此时如果保持坐标(X_{old},Y_{old})不变，而(W_{old},H_{old})减小，那么可能会导致猫的bounding box悬浮在空中。因此在这种情况下，
# 需要考虑输入的"Objects"字段，综合其他物体的位置，使得生成的猫的bounding box是与地面（或者其他东西）连接的，保证不会悬浮在空中（这是不和逻辑的）。

# I will give you the input consist of 3 fields named "Objects", "Old" and "New". The Input is as follow:\n\
#                         Objects: {[Name_1, (X_1,Y_1), (W_1,H_1))], [Name_2, (X_2,Y_2), (W_2,H_2))],... , [Name_n, (X_n,Y_n), (W_n,H_n))]}\n\
#                         Old: Name_{old}\nNew: Name_{new}\n\
#                         For the i-th item [Name_i, (X_i,Y_i), (W_i,H_i)] in the field "Objects", \
#                         Name_i represents its name (i.e. object class, such as cat, dog, apple and etc.), \
#                         and (X_i,Y_i), (W_i,H_i) represent the location and size respectively. \
#                         Additianally, (X_i,Y_i), (W_i,H_i) is in form of the bounding box, \
#                         where (X_i,Y_i) represent the coordinate of the point at the top left corner in the edge of bounding box, \
#                         And (W_i,H_i) represents the width and height of a rectangular box that including the i-th object; \
#                         Name_{old} in the "Old" field indicates an object to be removed in "Objects" field \
#                         (we ensure that Name_{old} must appear in "Objects" field), \
#                         while Name_{new} in the "New" field indicates a new object that will replace Name_{old}. \
#                         Additionally, The coordinate (X_{new}, Y_{new}) in output bounding box is just to fine-tune the position of object named Name_{new} \
#                         and it usually stays the same as input.  \


# system_prompt_replcace: 你是一个物体缩放器，能够根据一系列已知名字、已知大小的物体的信息，生成一个已知名字的物体的大小。“名字”即物体的类别，例如：猫，狗，苹果，等等。
#                         已知名字的物体的“大小”将用一个二元tuple (w,h)表示，这表示一个能够将物体包括在内的最小的矩形框的大小。另外，所有的物体其实在一张长方形画布（照片）中，
#                         所有矩形框（宽度和高度分别为w, h，且对宽度和高度的定义实际上符合opencv-python库的形式）。

# replace_first_ask: 针对你的任务，我会给你这样的输入：Objects: {[name_1, (w_1,h_1))], [name_2, (w_2,h_2))], ..., [name_2, (w_n,h_n))]}; New: name_{new}..
#                    其中"Objects"字段中[name_i, (w_i,h_i)]表示第i个物体的已知信息，name_i是物体的名字（类别），(w_i,h_i)表示包括第i个物体的矩形框的宽度和高度的大小；
#                    而“New”字段中name_{new}表示需要新增一个物体，你的工作就是根据已知名字(即name_i)的各种物体的大小(即w_i,h_i)来估计新增物体name_{new}的大小(e.i. w_{new}, h_{new})
#                    并输出为[name_{new}, (w_{new}, h_{new})]的形式。如果你已经完全明白你的任务，请回答“yes”，不要有任何多余的字符，在此之后我会向你给出输入。


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
                 'Second, if there is no clue of where to arange the new object, you should generate the '\
                 'location after understanding the instruction.\nAs is illustrated before, '\
                 'the location is in the format of quaternion as (X_i,Y_i,W_i,H_i) '\
                 '(representing a rectangle that crops the object), where X_i and Y_i represents the coordinates of '\
                 'the upper-left corner of the rectangle while W_i and H_i represents the width and height '\
                 'of the rectangle respectively. You should concatenate the name_i and the quaternion together '\
                 'to the format [name_i, (X_i,Y_i,W_i,H_i)] and add it to the object list (\'Image\'). '\
                 'Additionally, if the object mentioned in instruction is required to be removed, '\
                 'just delete it from "Image" when outputing.\nWhen it comes to the output, '\
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
                 'As a result, you only need to output a single word(\'<WHOLE>\' included, without quotation mark) without quotation mark.'\
                 'However, the answer you output mustn\'t contain any other character, only hte noun you found.'\
                 'If you have understood your task, answer \"yes\" without extra characters.'


import os

def get_bot(engine, api_key, system_prompt):
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
        print(f"talking {iteration}...... ", end='')
        try:
            answer = chatbot.ask(asks)
        except Exception as err:
            print(err)
            print('Timed Out', end='')
            if iteration > 2:
                time.sleep(5)
                os.system("bash ../clash/restart-clash.sh")
            time.sleep(5)
            if iteration % 5 == 4: print('')
            continue
        print('finish')
        # del chatbot
        return answer

system_prompt_expand =  'You are a prompt expander, expert in text expansion. Now you\'ll receive a prompt relative to '\
                        'text-driven image generation via Diffusion Models. That means your mission is to expand the '\
                        'prompt to much more elaborate one. The input will be a prompt starting with '\
                        '"a/an photo of ...", your task is to specify and expand the prompt to let diffusion make sense '\
                        'of the prompt so that diffusion can be driven to generate images with high quality and high resolution. '

two, three = 'two', 'three'
first_ask_expand = lambda x: 'For each of your input, it is ONLY ONE kind of object. '\
                        'For each input you received, you are only to output the expanded prompt without any other '\
                        'character. You mustn\'t output any extra characters except the expanded prompt. The expanded prompt is '\
                        f'ought to be no more than {two if x else three} sentences. If you\'ve '\
                        'made sense your task, please answer me \'yes\' and mustn\'t output any extra character, either, '\
                        'after which I\'ll give you input prompts. '
"""
    How many sentences generated by expander bot can affect the generation result of SDXL
    Experiment Result: No more than two sentences! (for the insufficient token)
"""





