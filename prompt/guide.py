from revChatGPT.V3 import Chatbot
import time, os
"""
    Remove, Replace 作为两个单独的状态，后面一个包括位置移动、大小修改、对象内容修改（ip2p）  【对象修改是否需要？还是我就做好位置、大小这些】
    
"""

system_prompt_sort = 'You are an expert in text classification,  and there are 5 classes in total.'\
                    '1. \"Remove\": determines whether the text removes the object, and if so, it is \"Romove\". '\
                    '2. \"Replace\": determine whether the text replaces the object, and if so, its category is \"Replace\". '\
                    '3. \"Move\": determine whether the text moves the object. If it does, the category is \"Move\". '\
                    '4. \"Add\": determine whether the text add several object. If it does, the category is \"Add\". '\
                    '5. \"Transfer\": determine whether the text is to do style transfering. If it does, the category is \"Transfer\". '\
                    'Note that the text is an editing instruction for the picture. We ensure all the text input is included in these 5 classes. \n'\
                    'For instance: \n'\
                    'Input: make the Ferris Wheel a giant hamster wheel\nOutput: \"Replace\"\n'\
                    'Input: make it an oil painting\nOutput: \"Transfer\"\n'\
                    'Input: have the ranch be a zoo\nOutput: \"Replace\"'
                        
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
                           'If you find object $A$ is replaced to object $B$, you are ought to give the answer $(A,B)$ without any other character or space. '\
                           'I guarantee only one object was replaced. '

replace_first_ask =     'You need to output both the replaced object $A$ and the replaced object $B$, in the form $(A,B)$. '\
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
system_prompt_locate =     'You are a text detector, expert at generating a new bounding box for a specific object. '\
                           'Inputs are in the form of the bellow: \n'\
                           'Size: $(W_{img},H{img})$\n'\
                           'Objects: ${[Name_1, (X_1,Y_1), (W_1,H_1))], [Name_2, (X_2,Y_2), (W_2,H_2))],..., '\
                           '[Name_n, (X_n,Y_n), (W_n,H_n))]}$\nTarget: Name_n\nEdit-Text: <text prompt>. \n'\
                           'As inputs from shown above, $(W_{img},H_{img})$ represents the size of original image input.'\
                           'And for the i-th item $[Name_i, (X_i,Y_i), (W_i,H_i)]$ in the field \"Objects\", '\
                           'Name_i labels its name (i.e. object class, such as cat, dog, apple and etc.), '\
                           'and $(X_i,Y_i), (W_i,H_i)$ represent the location and size respectively in an image(or a photo). '\
                           'Additianally, $(X_i,Y_i), (W_i,H_i)$ is in form of the bounding box, '\
                           'where $(X_i,Y_i)$ represent the coordinate of the point at the top left corner in the edge of bounding box, '\
                           'And $(W_i,H_i)$ represents the width and height of a rectangular box that including the i-th object. '\
                           '"Target" indicates a target to be edited, and we enssure '\
                           'that Name in \"Target\" field is equivalent to $Name_n$ in \"Objects\" field. '\
                           'Finally, in \"Edit-Text\" field, you will get the edit prompt. '\
                           'Your task is to arrange a proper place and size, in form of bounding box, '\
                           'For your task, you should output your generation of the target bounding box in the form of '\
                            '$[Name_n, (X_{new},Y_{new}), (W_{new}, H_{new})]$. $Name_n$ represents the name of the target '\
                            '(it stays the same!) and $[Name, (0,0), (0,0)]$ is strictly forbidden is your output. '\
                            'And your output mustn\'t contain any other character. '
                            
                        #    'for "Target" among objects in "Objects" field. '\
                        #    'For the coordinates designed in bounding box, we give relevant definitions. '\
                        #    'The upper left corner of a picture in the sense of human vision is the origin of coordinates; '\
                        #    'Starting from the origin, there are only two directions along the edge of the picture, "down" and "right". '\
                        #    'The direction "down" is defined as the positive direction of the y axis, '\
                        #    'and the direction "right" is defined as the positive direction of the x axis. '\
                        #    'In addition, for the width and height (i.e. $w$ and $h$) in bounding box, the former corresponds to the X-axis direction '\
                        #    'and the latter to the Y-axis direction. Therefore, given a bounding box quadtuple $((x,y), (w,h))$ '\
                        #    'corresponding to a rectangular region in a picture, the coordinates of the four vertices are '\
                        #    '$(x,y), (x+w,y), (w,y+h), (x+w,y+h)$, respectively. From this point of view, the values of $x+w$ and $y+h$ generated '\
                        #    'cannot exceed the size of the original image $(W_{img},H{img})$. Note that bounding boxes can overlap. '
          
locate_first_ask =      'If you have fully understood your task, '\
                        'please answer "yes" in the round without any extra characters. ' # replace the prompt befere
                        # 'For the coordinates designed in bounding box, we give relevant definitions. '\
                        # 'The upper left corner of a picture in the sense of human vision is the origin of coordinates; '\
                        # 'Starting from the origin, there are only two directions along the edge of the picture, "down" and "right". '\
                        # 'The direction "down" is defined as the positive direction of the y axis, '\
                        # 'and the direction "right" is defined as the positive direction of the x axis. '\
                        # 'In addition, for the width and height (i.e. $w$ and $h$) in bounding box, the former corresponds to the X-axis direction '\
                        # 'and the latter to the Y-axis direction. Therefore, given a bounding box quadtuple $((x,y), (w,h))$ '\
                        # 'corresponding to a rectangular region in a picture, the coordinates of the four vertices are '\
                        # '$(x,y), (x+w,y), (x,y+h), (x+w,y+h)$ respectively. From this point of view, the values of x+w and y+h generated '\
                        # 'cannot exceed the size of the original image $(W_{img},H{img})$.'\
                        # 'Your task is to modify the position and size of the target specified by the \"Target\" field ' \
                        # 'in the \"Objects\" field, according to the edit prompt given in \"Edit-Text\" field. '\
                        # 'You should arrange a proper position and a proper size for the editing target. '\
                        # 'In this way, your task is to generate the position and size according to the information '\
                        # 'given and representing them in the form of bounding box. '\
                        # 'For example, if the instruction in "Edit-Text" field told you to move a object to a far place, '\
                        # 'you should consider where is the so called "far place" according to all those objects given in "Objects" field, '\
                        # 'after which to generate a new bounding box which stands for its new position for the object.'\
                        

                    # TODO: consider if it's necessary to list the factors that GPT should take into account.
                    # TODO: Illustrate the generation procedure by providing examples ?

# <resize the object>
# no place
system_prompt_add = 'You are a text detection master and need to generate a new bounding box for a specific object. '\
                    'You are going to get a series texts input in the form of the bellow: \n'\
                    'Size: $(W_{img},H{img})$\n'\
                    'Objects: $\\{[Name_1, (X_1,Y_1), (W_1,H_1))], [Name_2, (X_2,Y_2), (W_2,H_2))],... , '\
                    '[Name_n, (X_n,Y_n), (W_n,H_n))]\\}$\nTarget: Name\n'\
                    'For the i-th item $[Name_i, (X_i,Y_i), (W_i,H_i)]$ in the field \"Objects\", '\
                    '$Name_i$ represents its name (i.e. object class, such as cat, dog, apple and etc.), '\
                    'and $(X_i,Y_i), (W_i,H_i)$ represent the location and size respectively in an image(or a photo). '\
                    'Additianally, $(X_i,Y_i), (W_i,H_i)$ is in form of the bounding box, '\
                    'where $(X_i,Y_i)$ represent the coordinate of the point at the top left corner in the edge of bounding box, '\
                    'and $(W_i,H_i)$ represents the width and height of a rectangular box that including the i-th object. '\
                    'You need to give the position and size of the specified new object "Name" in bounding box format. '\
                    'Regarding the arrangement of its position, you need to consider relative size to '\
                    'arrange a place $(X,Y)$ and a size $(W,H)$ for "Name". Your out put should be in format: '\
                    '$[Name, (X,Y,W,H)]$.'
# TODO: output Form ?

add_first_ask = 'If you have fully understand your task, please answer "yes" without any extra characters, and your output mustn\'t contain '\
                '$(0,0,0,0)$ as bounding box. '

system_prompt_addHelp = 'You will receive an instruction for image editing, which aims at adding objects '\
                        'to the image. Your task is to extract: What objects are to be added and How many respectively . '\
                        'Ensure that the input instruction contains only added operations and only one '\
                        '(but possibly multiple) objects. You need to output in the form (name, num, place), '\
                        'where \'name\' represents the kind of object to be added (for example, cat, dog, tiger), etc.'\
                        ', and place_n represents the target location to be added. '\
                        'Besides, if no place is specified, just put \"<NULL>\". By the way, the word like \'many\', \'some\', '\
                        'you can consider is as an adejuctive for \'name\', begining with \'some\', \'many\', etc. '\
                        'And simultaneously set \'num\'=1.'

addHelp_first_ask = 'For your task, in the output "name", note that you need to output the "name" modifier in the '\
                    'input instruction along with it. For example, if the input is "two more cats with black and '\
                    'white fur on the lawn," the output would be (" cats with black and white fur, "2," on the lawn "). '\
                    'In addition, your output must be of the form (name, num, place) and must not have any extra characters.'

system_prompt_addArrange = 'You are a text detection master and need to generate a new bounding box for a specific object. '\
                           'You should add an object to a specified place. '\
                           'You are going to get a series texts input in the form of the bellow: \n'\
                           'Size: $(W_{img},H{img})$\n'\
                           'Place: $[Name_p, (X_p,Y_p), (W_p,H_p)]$\nTarget: Name\n'\
                           'In the input shown above, $(W_{img},H_{img})$ in \"Size\" field represents the size of original image input. '\
                           'And int \"Place\" field the name of a scenery is given by $Name_p$, so as its location and range. '\
                           '\"Place\" is defined by $[Name_p, (X_p,Y_p), (W_p,H_p)]$ where $Name_p$ represents its name while '\
                           '$(X_p,Y_p)$ and $(W_p,H_p)$ represent its location and size in an image (or a photo). '\
                           'WHAT YOU SHOULD DO IS: arrange(generate) a proper location and size for the object \"Name\" '\
                           'with the constrain that the object \"Name\" represents is in the scope of \"Place\"'\
                           '$Name_p$ and $Name$ respectively represent a name (i.e. object class, such as \"cat\", \"dog\", \"apple\", etc.), '\
                           'and $(X_p,Y_p), (W_p,H_p)$ represent the location and size respectively in an image (or a photo) '\
                           'and a bounding box is uniquely identified by $(X_i,Y_i), (W_i,H_i)$ '\
                           'where $(X_p,Y_p)$ represents the coordinate of the point at the top left corner in the edge of bounding box, '\
                           'And $(W_p,H_p)$ represents the width and height of a rectangular box that including object. '\
                           'For the coordinates designed in bounding box, we give relevant definitions. '\
                           'The upper left corner of a picture in the sense of human vision is the origin of coordinates; '\
                           'Starting from the origin, there are only two directions along the edge of the picture, "down" and "right". '\
                           'The direction "down" is defined as the positive direction of the y axis, '\
                           'and the direction "right" is defined as the positive direction of the x axis. '\
                           'In addition, for the width and height (i.e. $w$ and $h$) in bounding box, the former corresponds to the X-axis direction '\
                           'and the latter to the Y-axis direction. Therefore, given a bounding box quadtuple $((x,y), (w,h))$ '\
                           'corresponding to a rectangular region in a picture, the coordinates of the four vertices are '\
                           '$(x,y), (x,y+h), (x+w,y), (x+w,y+h)$, respectively. From this point of view, the values of $x+w$ and $y+h$ generated '\
                           'cannot exceed the size of the original image $(W_{img},H{img})$. Note that bounding boxes can overlap. '

addArrange_first_ask =  'For the coordinates designed in bounding box, further definition is: '\
                        'The upper left corner of a picture in the sense of human vision is the origin of coordinates; '\
                        'Starting from the origin point, there are only two directions along the edge of the picture, "down" and "right". '\
                        'The direction "down" is defined as the positive direction of the y axis, '\
                        'and the direction "right" is defined as the positive direction of the x axis. '\
                        'In addition, for the width and height (i.e. $w$ and $h$) in bounding box, the former corresponds to the X-axis direction '\
                        'and the latter to the Y-axis direction. Therefore, given a bounding box quadtuple $((x,y), (w,h))$ '\
                        'corresponding to a rectangular region in a picture, the coordinates of the four vertices are '\
                        '$(x,y), (x+w,y), (x,y+h), (x+w,y+h)$, respectively. From this point of view, the values of $x+w$ and $y+h$ generated '\
                        'cannot exceed the size of the original image (W_{img},H{img}).'\
                        'Your task is to arrange the position and size of the target \"Name\" specified in the \"Target\" field ' \
                        'on purpose of adding it to the \"Place\" field. '\
                        'You should arrange a proper position and a proper size for the target. '\
                        'In this way, your task is to generate the position and size according to the information '\
                        'given and representing them in the form of bounding box. '\
                        'You should output your generation of the target bounding box in the form '\
                        '$[name_n, (X,Y), (W, H)]$ without any other character. $name_n$ represents the name of the target '\
                        '(it stays the same!). And coordinates $(X,Y)$ is the coordinate of the point at the top left corner in '\
                        'the edge of the bounding box, while $(W,H)$ represents the width and height of a '\
                        'rectangular box that including this object. Note that bounding boxes can overlap. '\
                        'If you have understood your task, '\
                        'please answer "yes" in the round without any extra characters, after which '\
                        'I will give you input.'

# 对于更加复杂的编辑任务，增加一个agent将指令分解为上述编辑单元

system_prompt_rescale =     'You are an object scaler, capable of generating a size and location for an object '\
                            'after considering size and location information of objects comprehensively. '\
                            'You\'ll be told a series of input messages in the form of as follow:\n '\
                            'Objects: $\\{[Name_1, (X_1,Y_1), (W_1,H_1))], [Name_2, (X_2,Y_2), (W_2,H_2))],... , [Name_n, (X_n,Y_n), (W_n,H_n))]\\}$\n'\
                            'Old: $Name_{old}$\nNew: $Name_{new}$\n'\
                            'For the i-th item $[Name_i, (X_i,Y_i), (W_i,H_i)]$ in the field "Objects", '\
                            '$Name_i$ represents its name (i.e. object class, such as cat, dog, apple and etc.), '\
                            'and $(X_i,Y_i), (W_i,H_i)$ represent the location and size respectively. '\
                            'Additianally, $(X_i,Y_i), (W_i,H_i)$ is in form of the bounding box, '\
                            'where $(X_i,Y_i)$ represent the coordinate of the point at the top left corner in the edge of bounding box, '\
                            'and $(W_i,H_i)$ represents the width and height of a rectangular box that including the i-th object. '\
                            'Then, in "Old" and "New" field , two nouns are given respectively, which indicates the $Name_{old}$ '\
                            'should be replaced $Name_{new}$. '\
                            'Based on the description above, your task is to generate a new position coordinates and sizes '\
                            'for the replacement. The out put should be in the form of $[Name_{new}, (X_{new},Y_{new}), (W_{new},H_{new})]$. '\
                            'Note that bounding boxes can overlap. '

rescale_first_ask =     'For your task, for details, you should generation modified bounding-box following the bellow rules. \n'\
                        '1. LOGIC. After the replacement is done, the objects must be placed logically. '\
                        'For example, if you need to replace a dog with a cat, usually the cat will be smaller than the '\
                        'dog (so the bounding box will be smaller), and if you keep the coordinates $(X_{old},Y_{old})$ '\
                        'unchanged, and $(W_{old},H_{old})$ decreases, then the cat\'s bounding box may be suspended in '\
                        'the air. So in this case, we need to take into account the input "Objects" field and combine '\
                        'the positions of the other objects, so that the generated bounding box of the cat is connected '\
                        'to the ground (or something else), ensuring that it will not be suspended in the air '\
                        '(which is illogical). \n'\
                        '2. STABILITY. As mentioned in the previous point, the position of the object needs to be '\
                        'modified after the replacement, but we still need to maintain the stability of its position, '\
                        'the modification needs to be integrated into the input of the "Objects" field in the position '\
                        'of each object in and overlay, size. In the case of guaranteeing requirement 1., '\
                        'if we can satisfy the logic stated in 1. without changing $X_{old}$ or $Y_{old}$, '\
                        'there\'s no necessity to change it. \n'\
                        'After the above two rules taken into consideration and finish the position and size editing, '\
                        'you should output the result in the form of $[Name_{new}, (X_{new},Y_{new}), (W_{new},H_{new})]$ '\
                        'without any other character. For each term I ask, you should only output the result in form of '\
                        '$[Name_{new}, (X_{new},Y_{new}), (W_{new}, H_{new})]$ and mustn\'t output any extra words. '\
                        'Now if you have fully understood your task, please answer "yes" and mustn\'t output any extra '\
                        'characters, after which I will give you input. '\

system_prompt_edit = 'You are an textual editor who is able to edit images with the given text input. '\
                     'But unlike traditional textual editors, you only need to edit the positions of some objects, '\
                     'which I will give in the following format: the i-th object is represented by the data '\
                     '$[name_i, (X_i,Y_i,W_i,H_i)]$. This format is consist of the following elements: '\
                     '1.$name_i$, namely a string representing the name of the i-th object, e.g. "apple" or "desk"; '\
                     '2. The quaternion $(X_i,Y_i,W_i,H_i)$, which is used to represent the rectangle cropping the i-th object '\
                     'and $(X_i,Y_i)$ represents the coordinates of the upper-left corner of the rectangle '\
                     'while $(W_i,H_i)$ represents the width and height of the rectangle, respectively. '\
                     'Thus, an image including N of objects is represented by $\\{(X_0,Y_0), [name_1,(X_1,Y_1,W_1,H_1)], '\
                     '[name_2, (X_2,Y_2,W_2,H_2)], ..., [name_N,(X_N,Y_N,W_N,H_N)]\\}$. '\
                     'Additionally, $(X_0,Y_0)$ at the beginning indicates the size of the image canvas, '\
                     'which is typically recognized as $(512, 512)$ unless the size is given.\n'\
                     'As I input such an image with a certain number of objects (in the format illustrated above), '\
                     'I\'ll bind all information in the format: "Instruction: ...;Image: ...;Size: ...", in which \'Size\' '\
                     'represents the size of the original image (if not given, the size is set to $512*512$) and \'Instruction\''\
                     ' is just the instruction, for example, \"move the apple to the right\", and \'Image\' is the'\
                     ' image formatted as above. What you need to do is to understand \'Instruction\' and output the \'Image\''\
                     ' in the same format as it input. For each item $[name_i,(X_i,Y_i,W_i,H_i)]$, '\
                     'you need to consider whether and how to modify the value of it in output according to the instruction '\
                     'in three aspects: 1. Name, namely $name_i$ in inputs. If the instruction specifies that the current '\
                     'name_i needs to be replaced by another object, then replace it with new object $name_i$ '\
                     'that menttioned in the instruction. 2. Position, indicated by both $X_i$ and $Y_i$ in inputs. '\
                     'If the instruction specifies that the current name_i corresponds to the object that needs to be '\
                     'moved to another place, follow it and amend $X_i$ and $Y_i$'\
                     '(If no position editing then keep them the same). 3. Size, if instruction specifies the '\
                     'size of an object named $name_i$ that needs to be changed (to increase it or decrease it), '\
                     'then modify the width and height of the rectangle box(namely $W_i$ and $H_i$); '\
                     'however, you shoulde ensure that the center of the rectangular box, noted as point $(x_i,y_i)$, '\
                     'which can be calculated by $x_i = X_i+0.5*W_i, y_i = Y_i+0.5*H_i$, stay the same. '\
                     'Note that if the instruction makes a request that the image canvas needs to be shrunk or magnified, '\
                     'you need to simultaneously decrease/increase the size of $W_i,H_i$ respectively. \n'\
                     'For inputs $\\{(X_0,Y_0), [name_1,(X_1,Y_1,W_1,H_1)], [name_2, (X_2,Y_2,W_2,H_2)], ... , '\
                     '[name_N,(X_N,Y_N,W_N,H_N)]\\}$, you only need to output $\\{(X_0\',Y_0\'), '\
                     '[name_1\',(X_1\',Y_1\',W_1\',H_1\')], [name_2\', (X_2\',Y_2\',W_2\',H_2\')], ..., '\
                     '[name_N\',(X_N\',Y_N\',W_N\',H_N\')]\\}$ and mustn\'t output any other character. '

first_ask_edit = 'Note that if you are following the instructions to do the modification, '\
                 'maybe some objects named $name_i$ which is not yet appeared before are added. '\
                 'For this case, the location to arange the newly added object depends on you. '\
                 'You should consider this issue according to the following aspects: '\
                 'First, if the location has been specified in the instruction, just follow it. '\
                 'Second, if there is no clue of where to arange the new object, you should generate the '\
                 'location after understanding the instruction.\nAs is illustrated before, '\
                 'the location is in the format of quaternion as $(X_i,Y_i,W_i,H_i)$ '\
                 '(representing a rectangle that crops the object), where $X_i$ and $Y_i$ represents the coordinates of '\
                 'the upper-left corner of the rectangle while $W_i$ and $H_i$ represents the width and height '\
                 'of the rectangle respectively. You should concatenate the $name_i$ and the quaternion together '\
                 'to the format $[name_i, (X_i,Y_i,W_i,H_i)]$ and add it to the object list (\'Image\'). '\
                 'Additionally, if the object mentioned in instruction is required to be removed, '\
                 'just delete it from "Image" when outputing.\nWhen it comes to the output, '\
                 'you should print three lines accoding to the description below: '\
                 'The first line is in the general output, in format: $\\{(X_0\',Y_0\'), [name_1\',(X_1\',Y_1\',W_1\',H_1\')]'\
                 ', [name_2\',(X_2\',Y_2\',W_1\',H_1\')], ..., [name_N\',(X_N\',Y_N\',W_N\',H_N\')]\\}$. '\
                 'namely the requirments illustrated before. The second line is new object list, in '\
                 'format: $\\{\'NEW\':[name_{k1}\',(X_{k1}\',Y_{k1}\',W_{k1}\',H_{k1}\')], ..., [name_{m1}\', '\
                 '(X_{m1}\',Y_{m1}\',W_{m1}\',H_{m1}\')]\\}$. It gathers all newly added objects in this line. '\
                 'The third line is disappeared object list, in format: $\\{\'DIS\':[name_{k2}\', '\
                 '(X_{k2}\',Y_{k2}\',W_{k2}\',H_{k2}\')], ..., [name_{m2}\',(X_{m2}\',Y_{m2}\',W_{m2}\',H_{m2}\')]\\}$. '\
                 'It gathers all objects removed in this line. However, there might be nothing newly added or nothing '\
                 'disappeared, so you print $\\{\'NEW\': NULL\\}$ or $\\{\'DIS\': NULL\\}$ instead in the coresponding lines.'\
                 'For example, I type to input --- Instruction: \'turn the cloud red.\'; Image: $\\{(500,388),[\'cloud\', '\
                 '(0,0,64,70)],[\'sun\',(400,0,64,64)]\\}$; Size: $(500,388)$. And you should give output: '\
                 '$\\{(500,388),[\'cloud\',(0,0,64,70)],[\'sun\',(400,0,64,64)]\\}\n\\{\'NEW\':NULL\\}\n\\{\'DIS\':NULL\\}$.\n\n'\
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

system_prompt_expand =  'You are a prompt expander, expert in text expansion. The input a prompt relative to '\
                        'text-driven image generation via Diffusion Models, starting with "a/an photo of ...". '\
                        'To enable the efficient generation, you are to expand the prompts. '\
                        'What you should do is to specify and expand the prompt to let diffusion model '\
                        'be driven to generate image with a clear outline. Your output (expanded prompts) '\
                        'must not be more than 1 sentence or contain any other character. '

one, two, three = 'one', 'two', 'three'
first_ask_expand = 'If you have fully understand your task, please answer \'yes\' without any other character. '
# first_ask_expand = lambda x: 'For each of your input, it is ONLY ONE kind of object. '\
#                         'For each input you received, you are only to output the expanded prompt without any other '\
#                         'character. You mustn\'t output any extra characters except the expanded prompt. The expanded prompt is '\
#                         f'ought to be no more than {one if x==1 else two if x==2 else three} sentences. If you\'ve '\
#                         'made sense your task, please answer me \'yes\' and mustn\'t output any extra character, either, '\
#                         'after which I\'ll give you input prompts. '
"""
    How many sentences generated by expander bot can affect the generation result of SDXL
    Experiment Result: No more than two sentences! (for the insufficient token)
"""

def get_bot(engine, api_key, system_prompt, proxy):
    iteration = 0
    while True:
        iteration += 1
        print(f"talking {iteration}......", end='')
        try:
            agent = Chatbot(engine=engine, api_key=api_key, system_prompt=system_prompt, proxy=proxy)
        except Exception as err:
            print('Error Msg: ', err)
            print('Apply Agent Timed Out')
            if iteration > 2:
                time.sleep(5)
                os.system("bash ../clash/restart-clash.sh")
            time.sleep(5)
            if iteration % 5 == 4: print('')
            continue
        print('Done')
        return agent

def get_response(chatbot, asks, mute_print=False):
    iteration = 0
    while True:
        iteration += 1
        if not mute_print: print(f"talking {iteration}...... ", end='')
        try:
            answer = chatbot.ask(asks)
        except Exception as err:
            string = f'Error Msg: {err}'
            print(string)
            print('Request Timed Out')
            if iteration > 2:
                os.system("bash ../clash/restart-clash.sh")
            time.sleep(50 if 'too many' in string.lower() else 10)
            if iteration % 5 == 4: print('')
            continue
        if not mute_print: print('Finish')
        return answer

def Use_Agent(opt, TODO=None, print_first_answer=False):
    # bounding box has its own engine
    if hasattr(opt, 'print_first_answer'): print_first_answer = opt.print_first_answer
    TODO = TODO.lower()
    engine, api_key, net_proxy = opt.engine, opt.api_key, opt.net_proxy
    box_engine = opt.box_engine if hasattr(opt, 'box_engine') else opt.engine
    if TODO == 'find target to be removed':
        agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_remove, proxy=net_proxy)
        first_ans = get_response(agent, remove_first_ask)
        if print_first_answer: print(first_ans) # first ask answer
    elif TODO == 'find target to be replaced':
        agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_replace, proxy=net_proxy)
        first_ans = get_response(agent, replace_first_ask)
        if print_first_answer: print(first_ans)
    elif TODO == 'rescale bbox for me': # Special Engine
        agent = get_bot(engine=box_engine, api_key=api_key, system_prompt=system_prompt_rescale, proxy=net_proxy)
        first_ans = get_response(agent, rescale_first_ask)
        if print_first_answer: print(first_ans)
    elif TODO == 'expand diffusion prompts for me':
        agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_expand, proxy=net_proxy)
        first_ans = get_response(agent, first_ask_expand)
        if print_first_answer: print(first_ans)
    elif TODO == 'arrange a new bbox for me': # Special Engine
        agent = get_bot(engine=box_engine, api_key=api_key, system_prompt=system_prompt_locate, proxy=net_proxy)
        first_ans = get_response(agent, locate_first_ask)
        if print_first_answer: print(first_ans)
    elif TODO == 'find target to be moved':
        agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_noun, proxy=net_proxy)
        first_ans = get_response(agent, first_ask_noun)
        if print_first_answer: print(first_ans)
    elif TODO == 'find target to be added':
        agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_addHelp, proxy=net_proxy)
        first_ans = get_response(agent, addHelp_first_ask)
        if print_first_answer: print(first_ans)
    elif TODO == 'generate a new bbox for me': # Special Engine
        agent = get_bot(engine=box_engine, api_key=api_key, system_prompt=system_prompt_add, proxy=net_proxy)
        first_ans = get_response(agent, add_first_ask)
        if print_first_answer: print(first_ans)
    elif TODO == 'adjust bbox for me': # Special Engine
        agent = get_bot(engine=box_engine, api_key=api_key, system_prompt=system_prompt_addArrange, proxy=net_proxy)
        first_ans = get_response(agent, addArrange_first_ask)
        if print_first_answer: print(first_ans)
    # elif TODO == 'use gpt-4v':
    #     agent = get_bot(engine=engine, api_key=api_key, system_prompt=gpt4_v_get_box, proxy=net_proxy)
    else:
        agent = None
        print('no such agent')
        exit(-1)
    print(f'Agent for \'{TODO}\' has been loaded')
    return agent

# for experiments

# 输入：cpation, label, bounding box
"""
    你是一个位置生成器，你需要根据输入的caption和label，为处在bounding box描述位置处的物体生成一个使用文字描述的位置。
    你获得的输入是：caption，label，（x,y,w,h）
    这里的(x,y,w,h)是bounding box，其含义为：(x,y)表示bounding box左上角的点的坐标，(w,h)为bounding box的宽度和高度。
    你需要将生成的描述性位置输出，同时给出另一个目标位置的文字描述，使得物体可以从当前位置被移动到目标位置，例如：
    Input: an apple is on the desk, apple, (100,100,50,70)
    Output: on the desk; under the desk
    Input: an apple on the desk, desk, (30,140,300,240)
    Output: desk on the left; desk on the right
    每个输出中的两个位置A和B用";"分割，你的输出禁止包含多余的无关字符
"""

system_prompt_gen_move_instructions = "You are a position generator and you need to generate a textual position " \
                                      "for an object at the position described by a bounding box, based on the input caption and label. " \
                                      "The input you get is: caption, label, (x,y,w,h). Here (x,y,w,h) is the bounding box, " \
                                      "which means: (x,y) represents the coordinates of the point in the upper left corner of " \
                                      "the bounding box, and (w,h) is the width and height of the bounding box. " \
                                      "You need to output the generated descriptive position with another textual description " \
                                      "of the target position, so that the object can be moved from its current position to the target position, for example: \n" \
                                      "Input: an apple is on the desk, apple, (100,100,50,70)\n" \
                                      "Output: on the desk; under the desk\n" \
                                      "Input: an apple on the desk, desk, (30,140,300,240)\n" \
                                      "Output: on the left; on the right\n" \
                                      "The two positions A and B in each output are separated by \";\", " \
                                      "and your output is forbidden to contain extra extraneous characters."

system_prompt_edit_sort = 'You are an expert in text classiffication,  and there are 5 classes in total.' \
                          '1. \"Remove\": determines whether the text removes the object, and if so, it is \"Romove\". ' \
                          '2. \"Replace\": determine whether the text replaces the object, and if so, its category is \"Replace\". ' \
                          '3. \"Move\": determine whether the text moves the object. If it does, the category is \"Move\". ' \
                          '4. \"Add\": determine whether the text add several object. If it does, the category is \"Add\". ' \
                          '5. \"Transfer\": determine whether the text is to do style transfering. If it does, the category is \"Transfer\". ' \
                          'Note that the text is an editing instruction for the picture. We ensure all the text input is included in these 5 classes. \n' \
                          'For instance: \n' \
                          'Input: make the Ferris Wheel a giant hamster wheel\nOutput: \"Replace\"\n' \
                          'Input: make it an oil painting\nOutput: \"Transfer\"\n' \
                          'Input: have the ranch be a zoo\nOutput: \"Replace\"\n' \
                          'Note that you are forbidden to include any other extra characters in your output.'

# not used yes

"""
    你是一个instruction生成器，你需要根据描述两幅相似图像的caption中的文字差异，生成一条能够通过“replace” 实现图像编辑的指令，例如：
    Input: 1.mountains of scotland under a bright sunny sky 2. mountains of scotland under a rainy sky
    Output: replace "bright sunny sky" with a "rainy sky".
    我们更希望你使用"replace A with B"的句型。另外，如果你认为这两条caption之间不能用一条只用“replace”方法的instruction实现，
    请输出“NULL”。例如：
    Input: 1. Aspen Country II Painting 2. Aspen Country II Cartoon
    Output: NULL
    因为这是一个风格迁移方面的变换
    注意，你的输出中禁止包含其他多余的字符。
"""

system_prompt_gen_replace_instructions = "You are an instruction generator, and you need to generate an instruction that "\
                                         "enables image editing via \"replace\" based on the textual differences in a "\
                                         "caption describing two similar images, for example: \n"\
                                         "Input: 1.mountains of scotland under a bright sunny sky 2. mountains of scotland under a rainy sky\n"\
                                         "Output: replace \"bright sunny sky\" with a \"rainy sky\". \n"\
                                         "We prefer you to use the \"replace A with B\" pattern. Also, if you think that the two captions "\
                                         "can't be separated by an instruction that only uses the \"replace\" method, just output \"NULL\". "\
                                         "For instance:\n"\
                                         "Input: 1. Aspen Country II Painting 2. Aspen Country II Cartoon\n"\
                                         "Output: NULL\nFor the fact that they are style transfering concerned instructions. "\
                                         "Note that you are forbidden to include any other extra characters in your output."

"""
    你是一个图像编辑系统，可以根编仅有的5个编辑工具给出编辑方案。
    你有且仅有以下5类工具做编辑: 'Add', 'Remove', 'Replace', 'Move' and 'Transfer'. 
    指令解释和效果如下所述。'Add'可以增加物体，例如可以实现"添加一个苹果", "在桌上放两个泰迪熊"；'
    Remove'可以去除物体，如可以用于"去掉桌上的梨子", "一个人把扫帚拿走了"；
    'Replace'用于替换物体，如"把狮子换成老虎", "把月亮换成太阳"；'Move'用于移动物体，
    如"把咖啡从电脑的左边拿到右边"；'Transfer'用于风格迁移，如"现代主义风格", 
    "转变成文艺复兴时期的风格". 对于输入的指令，需要你根据图像整体编辑要求给出编辑工具使用方案，
    并以$(type, method)$项的形式按顺序指明每一步的任务, 
    其中"type"是5种编辑工具中的一个(i.e. Add, Remove, Replace, Move, ransfer)
    而"method"表示实现的操作，即编辑工具的作用, 注意项与项之间以以";"分隔。
    以下是两个输入输出的例子。

    INPUT: a women enters the livingroom and take the box on the desk, while a cuckoo flies into the house.
    OUTPUT: (Remove, "remove the box on the desk"); (Add, "add a cukoo in the house")

    INPUT: "The sun went down, the sky was suddenly dark, and the birds returned to their nests."
    Output: (Remove, "remove the sun"); (Transfer, "the lights are out, darkness"); (Add, "add some birds, they are flying in the sky")
"""
planning_system_prompt = "You are an image editing system that can give editing solutions based on only 5 editing tools. "\
                            "You have and only have the following 5 types of tools for editing: 'Add', 'Remove', 'Replace', 'Move' and 'Transfer'. "\
                            "The commands are explained and their effects are described below. "\
                            "'Add' can add objects, such as \"Add an apple\", \"Put two teddy bears on the table\"; "\
                            "\'Add\' can add objects, such as \"add an apple \",\" put two teddy bears on the table \"; "\
                            "\'Remove\' can be used to remove objects, e.g. \'Remove a pear from a table\'; "\
                            "\'A person has taken the broom away\'; \'Replace\' is used to replace an object, "\
                            "such as \"replace a lion with a tiger \", \" replace the moon with the sun \"; "\
                            "\'Move\' is used to move something, as in \'move the coffee from the left side of "\
                            "the computer to the right side\'; \'Transfer\' is used for style transfer, e.g. "\
                            "\'modernist style\', \'to Renaissance style\'. For the input instructions, "\
                            "you need to give the editing tool use plan according to the overall editing requirements of the image. "\
                            "The tasks of each step are specified in order in the form of $(type, method)$item, "\
                            "where \"type\" is one of the five editing tools (i.e. Add, Remove, Replace, Move, ransfer) and \"method\" "\
                            "indicates the operation to be implemented. That is, the role of the editing tool, "\
                            "pay attention to the items between the \";\" Separate. Here are two examples of input and output. \n"\
                            "INPUT: a women enters the livingroom and take the box on the desk, while a cuckoo flies into the house. \n"\
                            "OUTPUT: (Remove, \"remove the box on the desk\");  (Add, \"add a cukoo in the house\"). \n\n"\
                            "INPUT: \"The sun went down, the sky was suddenly dark, and the birds returned to their nests. \"\n"\
                            "Output: (Remove, \"remove the sun\"); (Transfer, \"the lights are out, darkness\"); "\
                            "(Add, \"add some birds, they are flying in the sky\")\nNote that when you are giving output, \n"\
                            "A pair of parentheses with only a \"type\" and an \"edit instruction\", "\
                            "you mustn\'t output any other character"
planning_system_first_ask = "If you have understood your task, please answer \"yes\" without any other character and "\
                            "I\'ll give you the INPUT. Note that when you are giving output, you mustn\'t output any other character"


"""
    你是一个图片感知机，我需要你针对输入的图片做两件事情：1. 生成一条图片编辑指令； 2. 根据你设置的图片编辑指令设置一条询问这个编辑是否完成的一般疑问句。
    生成编辑指令时，编辑对象必须是图片中存在的物体，且指令必须在移动物体对象且带有方位信息，例如：“将苹果移动到桌子下面”，但前提是图片里有桌子。
    在生成疑问句时候，例如前面你生成的编辑指令是“将苹果移动到桌子下面”，那么生成的疑问句应当是：“苹果是不是在桌子下面？”，注意不要带有动词，尽量使用介词。
    请不要使用过度复杂的方位词信息，我们认为“将苹果移动到桌子下面”已经是一个足够复杂的句子了（请生成复杂程度相当的指令）。
    另外在生成时，编辑指令和疑问句各占一行，一共两行，禁止出现多余的字符。

"""

system_prompt_add_test = "You are a picture-aware machine, and I need you to do two things with an input picture: "\
                         "1. generate a picture editing instruction, and 2. set a general question asking if this edit is complete, "\
                         "based on the picture editing instruction you set. \nWhen generating an edit command, "\
                         "the object to be edited must be an object that exists in the picture, "\
                         "and the command must move the object with orientation information, for example: \"Move the apple under the table\""\
                         ", but only if there is a table in the picture. \nWhen generating a question, "\
                         "for example, if you generated an editorial instruction \"Move apples under the table\", "\
                         "then the question should be \"Are apples under the table?\". Be careful not to use verbs, "\
                         "and try to use prepositions. Please do not use overly complex orientation information. "\
                         "For instance, we think \"move the apple under the table\" is a complex enough sentence "\
                         "(please generate instructions of comparable complexity).\nNote that, when generating, the editorial instruction and "\
                         "the interrogative sentence each take up one line, totaling two lines, and superfluous characters are prohibited."

system_prompt_remove_test = "You are a picture-aware machine, and I need you to do two things with an input picture: "\
                         "1. generate a picture editing instruction, and 2. set a general question asking if this edit is complete, "\
                         "based on the picture editing instruction you set. \nWhen generating an edit command, "\
                         "the object to be edited must be an object that exists in the picture, "\
                         "and the command must move the object with orientation information, for example: \"Move the apple under the table\""\
                         ", but only if there is a table in the picture. \nWhen generating a question, "\
                         "for example, if you generated an editorial instruction \"Move apples under the table\", "\
                         "then the question should be \"Are apples under the table?\". Be careful not to use verbs, "\
                         "and try to use prepositions. Please do not use overly complex orientation information. "\
                         "For instance, we think \"move the apple under the table\" is a complex enough sentence "\
                         "(please generate instructions of comparable complexity).\nNote that, when generating, the editorial instruction and "\
                         "the interrogative sentence each take up one line, totaling two lines, and superfluous characters are prohibited."

# TODO: generate the complex task to be labeled
task_planning_test_system_prompt = ("You are a text generator. Your task is to generate image editing instructions" 
                                    "according to a input caption that describes the contents of a image. "
                                    "For edit instructions generation, the methods you can use are \"Add\" (for adding objects), "
                                    "\"Remove\" (for removing objects), \"Replace\" for (replacing the object with another), "
                                    "\"Move\" (for moving the object to another place), and \"Transfer\" (for image style transferring)."
                                    " What you need to do is to choose several methods to generate complex image editing instructions for the input image. "
                                    "You should output no more than 10 prompts (at least 3 prompts) for an image input. "
                                    "Note that your output should be an implicit edit command, for example, \"put the hat on the table\" "
                                    "is an implicit \"add\" command, \"the cat runs away\" is an implicit \"remove\" command, "
                                    "\"the cat runs under the desk\" is a \"move\" command. Each prompt is wrapped in parentheses, seperated by \"|\" "
                                    "Here's an example: \n"
                                    "INPUT: A tray filled with croissants with hotdogs in the middle.; A pastry displayed on a wood table in a store setting.; Troisgros are piled up for sale at a busy market. \n"
                                    "OUTPUT: (Hotdogs have already been eaten)|(pastry on the table creates)|(The desk is put to the corner)"
                                    "In this example, the edit method is \"Remove\", \"Add\", and \"Move\" respectively. Please use the object you are to edit as the subject. "
                                    "Note that your output should contain only the image editing instructions without any other characters.")



                 

